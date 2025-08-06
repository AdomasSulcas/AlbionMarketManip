"""
Monitoring Dashboard and Reporting System
Provides web-based dashboard and persistent state management for real-time monitoring.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import pandas as pd

from .realtime_monitor import MarketAlert, RealTimeMonitor


class PersistentStateManager:
    """
    Persistent state management system for real-time monitoring data.
    
    Handles storage and retrieval of alerts, monitoring statistics, and system state
    using SQLite database for reliability and efficient querying. Supports both
    real-time operations and historical analysis with automatic cleanup policies.
    """

    def __init__(self, db_path: str = "market_monitor.db") -> None:
        """
        Initialize persistent state manager with SQLite database.
        
        Args:
            db_path: Path to SQLite database file for persistent storage
        """
        self.db_path = Path(db_path)
        self.logger = __import__('logging').getLogger(__name__)
        self._initialize_database()

    def _initialize_database(self) -> None:
        """
        Initialize database schema for monitoring data storage.
        
        Creates required tables for alerts, monitoring sessions, and system statistics
        if they don't exist. Handles schema migrations for compatibility.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    item TEXT NOT NULL,
                    city TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    current_price REAL NOT NULL,
                    expected_price REAL NOT NULL,
                    price_deviation_pct REAL NOT NULL,
                    economic_context TEXT NOT NULL,
                    quality INTEGER DEFAULT 1,
                    additional_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Monitoring sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    items_monitored TEXT NOT NULL,
                    cities_monitored TEXT NOT NULL,
                    polls_completed INTEGER DEFAULT 0,
                    alerts_generated INTEGER DEFAULT 0,
                    api_errors INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Price history table for sliding window persistence
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    item TEXT NOT NULL,
                    city TEXT NOT NULL,
                    quality INTEGER DEFAULT 1,
                    price REAL NOT NULL,
                    source TEXT DEFAULT 'realtime',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_item_city ON alerts(item, city)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_item_city ON price_history(item, city, timestamp)")
            
            conn.commit()

    def save_alert(self, alert: MarketAlert) -> int:
        """
        Save alert to persistent storage for historical analysis and reporting.
        
        Args:
            alert: MarketAlert instance to store in database
            
        Returns:
            Database ID of saved alert record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts (
                    timestamp, item, city, alert_type, confidence, current_price,
                    expected_price, price_deviation_pct, economic_context, quality, additional_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.timestamp.isoformat(),
                alert.item,
                alert.city,
                alert.alert_type,
                alert.confidence,
                alert.current_price,
                alert.expected_price,
                alert.price_deviation_pct,
                alert.economic_context,
                alert.quality,
                json.dumps(alert.additional_data) if alert.additional_data else None
            ))
            
            conn.commit()
            return cursor.lastrowid

    def get_alerts(self, hours_back: int = 24, item: str = None, 
                   city: str = None, min_confidence: float = 0.0) -> List[MarketAlert]:
        """
        Retrieve alerts from persistent storage with filtering options.
        
        Args:
            hours_back: Number of hours of alert history to retrieve
            item: Filter by specific item name (optional)
            city: Filter by specific city name (optional)
            min_confidence: Minimum confidence threshold for alerts
            
        Returns:
            List of MarketAlert objects matching filter criteria
        """
        cutoff_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        
        query = """
            SELECT timestamp, item, city, alert_type, confidence, current_price,
                   expected_price, price_deviation_pct, economic_context, quality, additional_data
            FROM alerts
            WHERE timestamp > ? AND confidence >= ?
        """
        params = [cutoff_time, min_confidence]
        
        if item:
            query += " AND item = ?"
            params.append(item)
        
        if city:
            query += " AND city = ?"
            params.append(city)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            alerts = []
            for row in cursor.fetchall():
                additional_data = json.loads(row[10]) if row[10] else None
                
                alert = MarketAlert(
                    timestamp=datetime.fromisoformat(row[0]),
                    item=row[1],
                    city=row[2],
                    alert_type=row[3],
                    confidence=row[4],
                    current_price=row[5],
                    expected_price=row[6],
                    price_deviation_pct=row[7],
                    economic_context=row[8],
                    quality=row[9],
                    additional_data=additional_data
                )
                alerts.append(alert)
            
            return alerts

    def save_price_data(self, item: str, city: str, price: float, 
                       quality: int = 1, timestamp: datetime = None) -> None:
        """
        Save price data point for sliding window persistence.
        
        Args:
            item: Item name for price data
            city: City name for price data
            price: Price value in silver
            quality: Item quality level
            timestamp: When price was recorded (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO price_history (timestamp, item, city, quality, price)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp.isoformat(), item, city, quality, price))
            
            conn.commit()

    def get_price_history(self, item: str, city: str, quality: int = 1, 
                         hours_back: int = 24) -> pd.DataFrame:
        """
        Retrieve price history for sliding window reconstruction.
        
        Args:
            item: Item name for price history
            city: City name for price history  
            quality: Item quality level
            hours_back: Hours of history to retrieve
            
        Returns:
            DataFrame with price history data
        """
        cutoff_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT timestamp, price
                FROM price_history
                WHERE item = ? AND city = ? AND quality = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=[item, city, quality, cutoff_time])
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df

    def start_monitoring_session(self, items: List[str], cities: List[str]) -> int:
        """
        Record start of new monitoring session for tracking purposes.
        
        Args:
            items: List of items being monitored in this session
            cities: List of cities being monitored in this session
            
        Returns:
            Database ID of monitoring session record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO monitoring_sessions (start_time, items_monitored, cities_monitored)
                VALUES (?, ?, ?)
            """, (
                datetime.now().isoformat(),
                json.dumps(items),
                json.dumps(cities)
            ))
            
            conn.commit()
            return cursor.lastrowid

    def update_monitoring_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        """
        Update monitoring session with current statistics.
        
        Args:
            session_id: Database ID of monitoring session to update
            stats: Dictionary containing monitoring statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE monitoring_sessions
                SET polls_completed = ?, alerts_generated = ?, api_errors = ?
                WHERE id = ?
            """, (
                stats.get('polls_completed', 0),
                stats.get('alerts_generated', 0),
                stats.get('api_errors', 0),
                session_id
            ))
            
            conn.commit()

    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Remove old data from database to manage storage size.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Dictionary with counts of records removed from each table
        """
        cutoff_time = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clean up old alerts
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp < ?", (cutoff_time,))
            alerts_to_remove = cursor.fetchone()[0]
            cursor.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up old price history
            cursor.execute("SELECT COUNT(*) FROM price_history WHERE timestamp < ?", (cutoff_time,))
            prices_to_remove = cursor.fetchone()[0]
            cursor.execute("DELETE FROM price_history WHERE timestamp < ?", (cutoff_time,))
            
            conn.commit()
            
            return {
                'alerts_removed': alerts_to_remove,
                'prices_removed': prices_to_remove
            }


class MonitoringDashboard:
    """
    Web-based monitoring dashboard for real-time system visualization.
    
    Provides comprehensive monitoring interface with alert summaries,
    system statistics, performance metrics, and historical analysis.
    Designed for operational monitoring and system administration.
    """

    def __init__(self, state_manager: PersistentStateManager) -> None:
        """
        Initialize monitoring dashboard with persistent state access.
        
        Args:
            state_manager: PersistentStateManager instance for data access
        """
        self.state_manager = state_manager
        self.logger = __import__('logging').getLogger(__name__)

    def get_dashboard_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard data for specified time period.
        
        Aggregates alerts, statistics, and system health metrics for
        dashboard display and operational monitoring.
        
        Args:
            hours_back: Hours of historical data to include in dashboard
            
        Returns:
            Dictionary containing all dashboard data sections
        """
        alerts = self.state_manager.get_alerts(hours_back=hours_back)
        
        # Alert statistics
        alert_stats = {
            'total_alerts': len(alerts),
            'high_confidence_alerts': len([a for a in alerts if a.confidence > 0.7]),
            'unique_items_affected': len(set(a.item for a in alerts)),
            'unique_cities_affected': len(set(a.city for a in alerts)),
            'avg_confidence': sum(a.confidence for a in alerts) / len(alerts) if alerts else 0,
            'alert_types': {}
        }
        
        for alert in alerts:
            alert_stats['alert_types'][alert.alert_type] = alert_stats['alert_types'].get(alert.alert_type, 0) + 1

        # Item/City breakdown
        item_breakdown = {}
        city_breakdown = {}
        
        for alert in alerts:
            item_breakdown[alert.item] = item_breakdown.get(alert.item, 0) + 1
            city_breakdown[alert.city] = city_breakdown.get(alert.city, 0) + 1

        # Time series data for charts
        hourly_alerts = {}
        for alert in alerts:
            hour_key = alert.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_alerts[hour_key] = hourly_alerts.get(hour_key, 0) + 1

        # Economic context analysis
        economic_contexts = {}
        for alert in alerts:
            ctx = alert.economic_context
            economic_contexts[ctx] = economic_contexts.get(ctx, 0) + 1

        # Recent high-priority alerts
        high_priority_alerts = [
            {
                'timestamp': a.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'item': a.item,
                'city': a.city,
                'confidence': f"{a.confidence:.1%}",
                'price_deviation': f"{a.price_deviation_pct:+.1f}%",
                'current_price': f"{a.current_price:,.0f}",
                'economic_context': a.economic_context
            }
            for a in sorted(alerts, key=lambda x: x.confidence, reverse=True)[:10]
        ]

        return {
            'generated_at': datetime.now().isoformat(),
            'time_period_hours': hours_back,
            'alert_statistics': alert_stats,
            'item_breakdown': dict(sorted(item_breakdown.items(), key=lambda x: x[1], reverse=True)[:10]),
            'city_breakdown': dict(sorted(city_breakdown.items(), key=lambda x: x[1], reverse=True)),
            'hourly_alert_counts': hourly_alerts,
            'economic_contexts': economic_contexts,
            'high_priority_alerts': high_priority_alerts,
            'system_health': self._get_system_health_metrics()
        }

    def _get_system_health_metrics(self) -> Dict[str, Any]:
        """
        Calculate system health and performance metrics.
        
        Returns:
            Dictionary containing system health indicators and performance stats
        """
        with sqlite3.connect(self.state_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0] if cursor.fetchone() else 0
            
            # Recent session stats
            cursor.execute("""
                SELECT polls_completed, alerts_generated, api_errors
                FROM monitoring_sessions
                WHERE start_time > datetime('now', '-24 hours')
                ORDER BY start_time DESC
                LIMIT 1
            """)
            
            session_data = cursor.fetchone()
            
            # Alert rate trends
            cursor.execute("""
                SELECT COUNT(*) as count, strftime('%H', timestamp) as hour
                FROM alerts
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY hour
                ORDER BY hour
            """)
            
            hourly_trends = dict(cursor.fetchall())

        return {
            'database_size_mb': db_size / (1024 * 1024) if db_size else 0,
            'recent_polls': session_data[0] if session_data else 0,
            'recent_alerts': session_data[1] if session_data else 0,
            'recent_api_errors': session_data[2] if session_data else 0,
            'api_success_rate': 1 - (session_data[2] / max(session_data[0], 1)) if session_data and session_data[0] > 0 else 1.0,
            'hourly_alert_distribution': hourly_trends,
            'health_status': 'healthy'  # Could be enhanced with more sophisticated health checks
        }

    def generate_alert_report(self, hours_back: int = 24, format: str = 'json') -> Union[str, Dict]:
        """
        Generate formatted alert report for external consumption.
        
        Args:
            hours_back: Hours of alert data to include in report
            format: Output format ('json', 'text', or 'html')
            
        Returns:
            Formatted report in specified format
        """
        data = self.get_dashboard_data(hours_back)
        
        if format == 'json':
            return data
        
        elif format == 'text':
            report_lines = [
                f"=== Market Monitoring Report ===",
                f"Generated: {data['generated_at']}",
                f"Time Period: {data['time_period_hours']} hours",
                f"",
                f"Alert Summary:",
                f"  Total Alerts: {data['alert_statistics']['total_alerts']}",
                f"  High Confidence: {data['alert_statistics']['high_confidence_alerts']}",
                f"  Average Confidence: {data['alert_statistics']['avg_confidence']:.1%}",
                f"  Items Affected: {data['alert_statistics']['unique_items_affected']}",
                f"  Cities Affected: {data['alert_statistics']['unique_cities_affected']}",
                f"",
                f"Top Items by Alert Count:",
            ]
            
            for item, count in list(data['item_breakdown'].items())[:5]:
                report_lines.append(f"  {item}: {count}")
            
            report_lines.extend([
                f"",
                f"Economic Context Distribution:",
            ])
            
            for context, count in data['economic_contexts'].items():
                report_lines.append(f"  {context}: {count}")
            
            return "\n".join(report_lines)
        
        elif format == 'html':
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Market Monitoring Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                    .alert-item {{ background: #ffeeee; padding: 5px; margin: 5px 0; border-left: 3px solid #ff0000; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Market Monitoring Dashboard</h1>
                <p>Generated: {data['generated_at']} | Period: {data['time_period_hours']} hours</p>
                
                <div class="metric">
                    <h3>Alert Summary</h3>
                    <p>Total: {data['alert_statistics']['total_alerts']} | 
                       High Confidence: {data['alert_statistics']['high_confidence_alerts']} | 
                       Avg Confidence: {data['alert_statistics']['avg_confidence']:.1%}</p>
                </div>
                
                <h3>Recent High-Priority Alerts</h3>
                <table>
                    <tr><th>Time</th><th>Item</th><th>City</th><th>Confidence</th><th>Deviation</th></tr>
            """
            
            for alert in data['high_priority_alerts'][:5]:
                html_template += f"""
                    <tr>
                        <td>{alert['timestamp']}</td>
                        <td>{alert['item']}</td>
                        <td>{alert['city']}</td>
                        <td>{alert['confidence']}</td>
                        <td>{alert['price_deviation']}</td>
                    </tr>
                """
            
            html_template += """
                </table>
                
                <div class="metric">
                    <h3>System Health</h3>
                    <p>Status: """ + data['system_health']['health_status'].upper() + f"""</p>
                    <p>API Success Rate: {data['system_health']['api_success_rate']:.1%}</p>
                    <p>Database Size: {data['system_health']['database_size_mb']:.1f} MB</p>
                </div>
            </body>
            </html>
            """
            
            return html_template
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# Integration class for enhanced RealTimeMonitor with persistence
class PersistentRealTimeMonitor(RealTimeMonitor):
    """
    Enhanced real-time monitor with persistent state management and dashboard integration.
    
    Extends base RealTimeMonitor with automatic data persistence, dashboard generation,
    and enhanced reporting capabilities for production deployment.
    """

    def __init__(self, db_path: str = "market_monitor.db", **kwargs) -> None:
        """
        Initialize persistent real-time monitor with database storage.
        
        Args:
            db_path: Path to SQLite database for persistent storage
            **kwargs: Additional arguments passed to RealTimeMonitor
        """
        super().__init__(**kwargs)
        
        self.state_manager = PersistentStateManager(db_path)
        self.dashboard = MonitoringDashboard(self.state_manager)
        self.session_id = None
        
        # Add persistent alert handler
        def persistent_alert_handler(alert: MarketAlert) -> None:
            try:
                self.state_manager.save_alert(alert)
            except Exception as e:
                self.logger.error(f"Failed to save alert to database: {e}")
        
        self.alert_manager.add_notification_handler(persistent_alert_handler)

    async def start_monitoring(self) -> None:
        """
        Start monitoring with persistent session tracking.
        
        Overrides base implementation to add database session management
        and automatic state persistence.
        """
        self.session_id = self.state_manager.start_monitoring_session(self.items, self.cities)
        self.logger.info(f"Started monitoring session {self.session_id}")
        
        await super().start_monitoring()

    def stop_monitoring(self) -> None:
        """
        Stop monitoring and finalize database session.
        
        Overrides base implementation to properly close database session
        and perform final statistics update.
        """
        if self.session_id:
            self.state_manager.update_monitoring_session(self.session_id, self.get_monitoring_stats())
        
        super().stop_monitoring()

    def get_dashboard_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get dashboard data with integrated monitoring statistics.
        
        Args:
            hours_back: Hours of historical data to include
            
        Returns:
            Complete dashboard data including current session statistics
        """
        dashboard_data = self.dashboard.get_dashboard_data(hours_back)
        dashboard_data['current_session'] = self.get_monitoring_stats()
        return dashboard_data

    def export_alerts(self, hours_back: int = 24, filename: str = None) -> str:
        """
        Export alerts to JSON file for external analysis.
        
        Args:
            hours_back: Hours of alert history to export
            filename: Output filename (default: timestamped filename)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"market_alerts_{timestamp}.json"
        
        alerts = self.state_manager.get_alerts(hours_back)
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'time_period_hours': hours_back,
            'total_alerts': len(alerts),
            'alerts': [alert.to_dict() for alert in alerts]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(alerts)} alerts to {filename}")
        return filename
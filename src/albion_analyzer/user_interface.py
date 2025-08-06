"""
Market Manipulation Detection Output Interface
Provides both professional fraud detection reports and user-friendly market safety alerts.
"""

from typing import Dict, List, Any
from datetime import datetime
import logging
import csv
import os
from pathlib import Path


class FraudDetectionReporter:
    """
    Professional fraud detection reporting for market manipulation analysis.
    
    Provides detailed manipulation evidence, severity assessment, and pattern analysis
    for investigators, traders, and market oversight professionals.
    """
    
    def __init__(self):
        """Initialize fraud detection reporter."""
        self.logger = logging.getLogger(__name__)
    
    def generate_fraud_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive fraud detection report for professionals.
        
        Args:
            results: Analysis results from market manipulation detection
            
        Returns:
            Detailed fraud analysis report
        """
        if 'error' in results:
            return f"[FRAUD ANALYSIS ERROR] {results['error']}"
        
        anomalies = results.get('anomalies', [])
        stats = results.get('processing_stats', {})
        
        if not anomalies:
            return self._generate_clean_market_fraud_report(stats)
        
        return self._generate_manipulation_fraud_report(anomalies, stats)
    
    def _generate_clean_market_fraud_report(self, stats: Dict[str, Any]) -> str:
        """Generate fraud report when no manipulation detected."""
        items_checked = stats.get('items_processed', 0)
        
        report = [
            "",
            "=== MARKET MANIPULATION FRAUD ANALYSIS ===",
            "",
            "FRAUD STATUS: NO MANIPULATION DETECTED",
            f"Items Analyzed: {items_checked}",
            f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "FINDINGS:",
            "• No evidence of coordinated market manipulation",
            "• Current prices align with historical baselines",
            "• No suspicious buyout patterns detected",
            "• Market appears to be operating under normal conditions",
            "",
            "RECOMMENDATION: Market monitoring should continue at normal intervals",
            ""
        ]
        
        return "\n".join(report)
    
    def _generate_manipulation_fraud_report(self, anomalies: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
        """Generate detailed fraud report when manipulation detected."""
        items_checked = stats.get('items_processed', 0)
        
        # Categorize by fraud severity
        confirmed_fraud = [a for a in anomalies if a['price_deviation_pct'] > 1000]  # >10x price
        likely_fraud = [a for a in anomalies if 500 < a['price_deviation_pct'] <= 1000]  # 5-10x price
        suspicious_activity = [a for a in anomalies if 100 < a['price_deviation_pct'] <= 500]  # 2-5x price
        
        report = [
            "",
            "=== MARKET MANIPULATION FRAUD ANALYSIS ===",
            "",
            "FRAUD STATUS: MANIPULATION DETECTED",
            f"Items Analyzed: {items_checked}",
            f"Anomalies Found: {len(anomalies)}",
            f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if confirmed_fraud:
            report.extend([
                f"[CONFIRMED FRAUD] {len(confirmed_fraud)} items with extreme manipulation evidence:",
                ""
            ])
            for fraud in confirmed_fraud[:5]:
                report.extend(self._format_fraud_evidence(fraud, "CONFIRMED"))
        
        if likely_fraud:
            report.extend([
                f"[LIKELY FRAUD] {len(likely_fraud)} items with strong manipulation indicators:",
                ""
            ])
            for fraud in likely_fraud[:5]:
                report.extend(self._format_fraud_evidence(fraud, "LIKELY"))
        
        if suspicious_activity:
            report.extend([
                f"[SUSPICIOUS] {len(suspicious_activity)} items requiring investigation:",
                ""
            ])
            for fraud in suspicious_activity[:3]:
                report.extend(self._format_fraud_evidence(fraud, "SUSPICIOUS"))
        
        # Add pattern analysis
        report.extend([
            "MANIPULATION PATTERN ANALYSIS:",
            self._analyze_manipulation_patterns(anomalies),
            "",
            "RECOMMENDED ACTIONS:",
            "• Flag identified items for enhanced monitoring",
            "• Investigate unusual trading volumes in affected markets",
            "• Cross-reference with large transaction logs",
            "• Monitor for coordinated activity across multiple items",
            ""
        ])
        
        return "\n".join(report)
    
    def _format_fraud_evidence(self, anomaly: Dict[str, Any], severity: str) -> List[str]:
        """Format individual fraud evidence entry."""
        item = anomaly['item']
        city = anomaly['city']
        current_price = anomaly['current_price']
        baseline = anomaly['baseline_median']
        deviation = anomaly['price_deviation_pct']
        confidence = anomaly['confidence']
        
        multiplier = (current_price / baseline) if baseline > 0 else 0
        
        evidence = [
            f"  • {item} in {city}",
            f"    Current Price: {current_price:,} silver",
            f"    Historical Baseline: {baseline:,.0f} silver",
            f"    Price Inflation: {deviation:+.0f}% ({multiplier:.1f}x normal price)",
            f"    Fraud Confidence: {confidence:.2f}",
            f"    Evidence: {', '.join(anomaly['anomaly_reasons'][:2])}",
            ""
        ]
        
        return evidence
    
    def _analyze_manipulation_patterns(self, anomalies: List[Dict[str, Any]]) -> str:
        """Analyze patterns in manipulation attempts."""
        if not anomalies:
            return "No patterns detected"
        
        # Analyze by city
        city_counts = {}
        for anomaly in anomalies:
            city = anomaly['city']
            city_counts[city] = city_counts.get(city, 0) + 1
        
        # Analyze by item type
        item_patterns = {}
        for anomaly in anomalies:
            item = anomaly['item']
            if '_' in item:
                item_type = '_'.join(item.split('_')[1:])  # Remove tier prefix
                item_patterns[item_type] = item_patterns.get(item_type, 0) + 1
        
        patterns = []
        
        # City concentration
        if city_counts:
            most_affected = max(city_counts, key=city_counts.get)
            patterns.append(f"Most affected city: {most_affected} ({city_counts[most_affected]} items)")
        
        # Item type patterns
        if item_patterns:
            top_category = max(item_patterns, key=item_patterns.get)
            patterns.append(f"Most targeted category: {top_category} ({item_patterns[top_category]} items)")
        
        # Severity distribution
        extreme_count = len([a for a in anomalies if a['price_deviation_pct'] > 500])
        if extreme_count > 0:
            patterns.append(f"Extreme manipulations: {extreme_count}/{len(anomalies)} cases")
        
        return "• " + "\n• ".join(patterns) if patterns else "No significant patterns detected"


class ItemNameTranslator:
    """
    Translates technical item codes to human-readable names.
    
    Converts API item codes like "T6_2H_AXE" to user-friendly names
    like "Tier 6 Two-Handed Axe" for better user experience.
    """
    
    def __init__(self):
        """Initialize item name mappings."""
        self.tier_names = {
            'T3': 'Tier 3', 'T4': 'Tier 4', 'T5': 'Tier 5',
            'T6': 'Tier 6', 'T7': 'Tier 7', 'T8': 'Tier 8'
        }
        
        self.item_types = {
            # Weapons
            '2H_AXE': 'Two-Handed Axe',
            '2H_BOW': 'Two-Handed Bow', 
            '2H_SWORD': 'Two-Handed Sword',
            '2H_MACE': 'Two-Handed Mace',
            '2H_HAMMER': 'Two-Handed Hammer',
            '2H_CROSSBOW': 'Two-Handed Crossbow',
            'MAIN_AXE': 'One-Handed Axe',
            'MAIN_BOW': 'One-Handed Bow',
            'MAIN_SWORD': 'One-Handed Sword',
            'MAIN_MACE': 'One-Handed Mace',
            'MAIN_HAMMER': 'One-Handed Hammer',
            'MAIN_CROSSBOW': 'One-Handed Crossbow',
            
            # Staves
            '2H_ARCANESTAFF': 'Arcane Staff',
            '2H_CURSEDSTAFF': 'Cursed Staff',
            '2H_FIRESTAFF': 'Fire Staff',
            '2H_FROSTSTAFF': 'Frost Staff',
            '2H_HOLYSTAFF': 'Holy Staff',
            '2H_NATURESTAFF': 'Nature Staff',
            'MAIN_ARCANESTAFF': 'One-Handed Arcane Staff',
            'MAIN_CURSEDSTAFF': 'One-Handed Cursed Staff',
            'MAIN_FIRESTAFF': 'One-Handed Fire Staff',
            'MAIN_FROSTSTAFF': 'One-Handed Frost Staff',
            'MAIN_HOLYSTAFF': 'One-Handed Holy Staff',
            'MAIN_NATURESTAFF': 'One-Handed Nature Staff',
            
            # Armor
            'HEAD_PLATE_SET1': 'Plate Helmet',
            'ARMOR_PLATE_SET1': 'Plate Armor',
            'SHOES_PLATE_SET1': 'Plate Boots',
            'HEAD_LEATHER_SET1': 'Leather Hood',
            'ARMOR_LEATHER_SET1': 'Leather Armor', 
            'SHOES_LEATHER_SET1': 'Leather Shoes',
            'HEAD_CLOTH_SET1': 'Cloth Cowl',
            'ARMOR_CLOTH_SET1': 'Cloth Robe',
            'SHOES_CLOTH_SET1': 'Cloth Sandals',
            
            # Accessories
            'BAG': 'Bag',
            'CAPE': 'Cape',
            
            # Mounts
            'MOUNT_HORSE': 'Riding Horse',
            'MOUNT_HORSE_UNDEAD': 'Undead Horse',
            'MOUNT_OX': 'Transport Ox',
            'MOUNT_STAG': 'Stag',
            'MOUNT_SWIFTCLAW': 'Swiftclaw',
            'MOUNT_BEAR': 'Bear',
            'MOUNT_DIREWOLF': 'Direwolf',
            
            # Resources
            'ORE': 'Ore',
            'HIDE': 'Hide',
            'FIBER': 'Fiber',
            'WOOD': 'Wood',
            'STONE': 'Stone',
            'CLOTH': 'Cloth',
            'METALBAR': 'Metal Bar',
            'PLANKS': 'Planks',
            'STONEBLOCK': 'Stone Block',
            'LEATHER': 'Leather',
            
            # Food
            'MEAL_PIE': 'Pie',
            'MEAL_SOUP': 'Soup',
            'MEAL_BREAD': 'Bread',
            'MEAL_SANDWICH': 'Sandwich',
            'MEAL_SALAD': 'Salad',
            'MEAL_FISH': 'Cooked Fish',
            'MEAL_OMELETTE': 'Omelette',
            
            # Potions
            'POTION_HEAL': 'Healing Potion',
            'POTION_ENERGY': 'Energy Potion',
            'POTION_STONESKIN': 'Stoneskin Potion',
            'POTION_RESISTANCE': 'Resistance Potion',
            'POTION_STICKY': 'Sticky Potion',
            'POTION_SLOW': 'Slow Potion',
            
            # Tools
            'TOOL_PICKAXE': 'Pickaxe',
            'TOOL_AXE': 'Axe',
            'TOOL_HAMMER': 'Hammer', 
            'TOOL_HOE': 'Hoe',
            'TOOL_KNIFE': 'Knife',
            'TOOL_SICKLE': 'Sickle',
            'TOOL_SKINNING_KNIFE': 'Skinning Knife',
            'TOOL_NEEDLE': 'Needle'
        }

    def translate_item_name(self, item_code: str) -> str:
        """
        Convert technical item code to human-readable name.
        
        Args:
            item_code: Technical item code like "T6_2H_AXE"
            
        Returns:
            Human-readable name like "Tier 6 Two-Handed Axe"
        """
        if not item_code:
            return "Unknown Item"
        
        parts = item_code.split('_')
        if len(parts) < 2:
            return item_code
        
        tier = parts[0]
        item_type = '_'.join(parts[1:])
        
        tier_name = self.tier_names.get(tier, tier)
        type_name = self.item_types.get(item_type, item_type.replace('_', ' ').title())
        
        return f"{tier_name} {type_name}"


class MarketSafetyReporter:
    """
    Generates market safety alerts for regular users.
    
    Focuses on protecting regular users from market manipulation by providing
    clear warnings about overpriced items and safe trading recommendations.
    """

    def __init__(self):
        """Initialize market safety reporter."""
        self.translator = ItemNameTranslator()
        self.logger = logging.getLogger(__name__)

    def generate_market_safety_alert(self, results: Dict[str, Any]) -> str:
        """
        Generate market safety alert for regular users.
        
        Args:
            results: Analysis results from fraud detection
            
        Returns:
            Clear safety alert with actionable recommendations
        """
        if 'error' in results:
            return f"⚠️ Market analysis unavailable: {results['error']}"
        
        anomalies = results.get('anomalies', [])
        stats = results.get('processing_stats', {})
        
        if not anomalies:
            return self._generate_safe_market_alert(stats)
        
        return self._generate_manipulation_safety_alert(anomalies, stats)

    def _generate_safe_market_alert(self, stats: Dict[str, Any]) -> str:
        """Generate alert when market is safe from manipulation."""
        items_checked = stats.get('items_processed', 0)
        
        alert = [
            "",
            "[SAFE] MARKET SAFETY: ALL CLEAR",
            f"[OK] Analyzed {items_checked} items - no manipulation detected",
            "[OK] Prices appear normal and fair",
            "[OK] Safe to buy and sell items at current market rates",
            "",
            "[TIP] Market conditions are healthy for regular trading",
            "[TIME] Check again in 30-60 minutes for updates",
            ""
        ]
        
        return "\n".join(alert)

    def _generate_manipulation_safety_alert(self, anomalies: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
        """Generate safety alert when manipulation detected."""
        items_checked = stats.get('items_processed', 0)
        
        # Categorize by danger level for users
        dangerous_items = [a for a in anomalies if a['price_deviation_pct'] > 500]  # >5x price - clear fraud
        overpriced_items = [a for a in anomalies if 100 < a['price_deviation_pct'] <= 500]  # 2-5x price
        suspicious_items = [a for a in anomalies if 30 < a['price_deviation_pct'] <= 100]  # 30-100% higher
        
        alert = [
            "",
            "[WARNING] MARKET MANIPULATION DETECTED",
            f"[SCAN] Checked {items_checked} items - Found {len(anomalies)} suspicious prices",
            "",
            "[ALERT] SOMEONE IS MANIPULATING ITEM PRICES!",
            ""
        ]
        
        if dangerous_items:
            alert.extend([
                "[DANGER] DO NOT BUY - OBVIOUS FRAUD:",
                ""
            ])
            for item in dangerous_items[:5]:
                item_name = self.translator.translate_item_name(item['item'])
                current = item['current_price']
                baseline = item['baseline_median']
                city = item['city']
                multiplier = current / baseline if baseline > 0 else 0
                
                alert.extend([
                    f"   [FRAUD] {item_name} in {city}",
                    f"      Price: {current:,} silver (Normal: {baseline:,.0f})",
                    f"      [SCAM] {multiplier:.1f}x MORE EXPENSIVE than normal - FRAUD!",
                    ""
                ])
        
        if overpriced_items:
            alert.extend([
                "[CAUTION] AVOID THESE - SEVERELY OVERPRICED:",
                ""
            ])
            for item in overpriced_items[:3]:
                item_name = self.translator.translate_item_name(item['item'])
                current = item['current_price']
                deviation = item['price_deviation_pct']
                city = item['city']
                
                alert.extend([
                    f"   [AVOID] {item_name} in {city}",
                    f"      Price: {current:,} silver (+{deviation:.0f}% overpriced)",
                    f"      [TIP] Wait or check other cities",
                    ""
                ])
        
        if suspicious_items:
            remaining = len(suspicious_items)
            alert.extend([
                f"[WATCH] BE CAREFUL ({remaining} more items moderately overpriced)",
                "   Consider checking other cities or waiting",
                ""
            ])
        
        # Safety recommendations
        alert.extend([
            "[SAFETY] STAY SAFE:",
            "   • DO NOT buy items marked [FRAUD] - they are scams",
            "   • [AVOID] items unless desperate - try other cities first",
            "   • Items NOT listed here should have normal prices",
            "   • Check back in 30-60 minutes - manipulation often ends",
            "   • Report suspected manipulation to game admins",
            "",
            f"[TIME] Last checked: {datetime.now().strftime('%H:%M:%S')} - Check again soon",
            ""
        ])
        
        return "\n".join(alert)

    def generate_quick_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate ultra-quick one-line summary for busy users.
        
        Args:
            results: Analysis results
            
        Returns:
            One-line summary
        """
        anomalies = results.get('anomalies', [])
        
        if not anomalies:
            return "🟢 Market is healthy - no manipulation detected"
        
        extreme = len([a for a in anomalies if a['price_deviation_pct'] > 500])
        high = len([a for a in anomalies if 100 < a['price_deviation_pct'] <= 500])
        
        if extreme > 0:
            return f"🔴 {extreme} items EXTREMELY overpriced - avoid buying these!"
        elif high > 0:
            return f"🟡 {high} items significantly overpriced - check other cities"
        else:
            return f"🟠 {len(anomalies)} items moderately overpriced - mostly normal market"

    def generate_trader_report(self, results: Dict[str, Any]) -> str:
        """
        Generate focused report for traders looking for opportunities.
        
        Args:
            results: Analysis results
            
        Returns:
            Trader-focused report with opportunities
        """
        anomalies = results.get('anomalies', [])
        
        if not anomalies:
            return self._generate_trader_normal_market()
        
        return self._generate_trader_opportunities(anomalies)

    def _generate_trader_normal_market(self) -> str:
        """Generate trader report for normal market conditions."""
        return """
💰 TRADER REPORT - NORMAL MARKET CONDITIONS

🟢 Market Status: Healthy - No Major Manipulation
✓ Good time for normal trading activities
✓ Prices are stable and predictable
✓ Low risk of buying into artificial inflation

📈 Trading Opportunities:
• Normal arbitrage between cities should work well  
• Item flipping at standard margins
• Safe to stock up on materials at current prices
• Focus on natural supply/demand differences

⚠️  Stay alert - run this scan every 30-60 minutes to catch new manipulation
"""

    def _generate_trader_opportunities(self, anomalies: List[Dict[str, Any]]) -> str:
        """Generate trader report when anomalies exist."""
        # Separate into opportunities vs risks
        avoid_list = [a for a in anomalies if a['price_deviation_pct'] > 100]
        caution_list = [a for a in anomalies if 30 < a['price_deviation_pct'] <= 100]
        
        report = [
            "",
            "💰 TRADER REPORT - MANIPULATION DETECTED",
            "",
            "🚫 AVOID BUYING (Artificial Inflation):"
        ]
        
        for alert in avoid_list[:5]:
            item_name = self.translator.translate_item_name(alert['item'])
            current = alert['current_price']
            city = alert['city']
            deviation = alert['price_deviation_pct']
            
            report.append(f"   • {item_name} in {city}")
            report.append(f"     {current:,.0f} silver (+{deviation:.0f}%) - Wait for crash")
        
        if not avoid_list:
            report.append("   • None identified - relatively safe market")
        
        report.extend([
            "",
            "⚠️  EXERCISE CAUTION:"
        ])
        
        for alert in caution_list[:3]:
            item_name = self.translator.translate_item_name(alert['item'])
            city = alert['city']
            deviation = alert['price_deviation_pct']
            
            report.append(f"   • {item_name} in {city} (+{deviation:.0f}%)")
        
        if not caution_list:
            report.append("   • Market relatively stable for trading")
        
        report.extend([
            "",
            "💡 TRADER STRATEGIES:",
            "   • Short-term: Avoid manipulated items entirely",
            "   • Medium-term: Watch for price crashes on avoided items",
            "   • Arbitrage: Focus on items NOT on these lists",
            "   • Safe bet: Trade in cities not showing manipulation",
            ""
        ])
        
        return "\n".join(report)


def format_fraud_detection_report(results: Dict[str, Any]) -> str:
    """
    Format professional fraud detection report.
    
    Args:
        results: Analysis results from market manipulation detection
        
    Returns:
        Professional fraud analysis report
    """
    reporter = FraudDetectionReporter()
    return reporter.generate_fraud_report(results)

def format_market_safety_alert(results: Dict[str, Any]) -> str:
    """
    Format market safety alert for regular users.
    
    Args:
        results: Analysis results from fraud detection
        
    Returns:
        User-friendly market safety alert
    """
    reporter = MarketSafetyReporter()
    return reporter.generate_market_safety_alert(results)


class CSVFraudReporter:
    """
    Exports fraud detection results to structured CSV files for analysis.
    
    Creates separate CSV files for different fraud severity levels:
    - confirmed_fraud.csv (>1000% price manipulation)
    - likely_fraud.csv (500-1000% price manipulation)  
    - suspicious_activity.csv (100-500% price manipulation)
    - moderate_overpricing.csv (30-100% price manipulation)
    """
    
    def __init__(self):
        """Initialize CSV fraud reporter."""
        self.translator = ItemNameTranslator()
        self.logger = logging.getLogger(__name__)
    
    def export_fraud_analysis_csv(self, results: Dict[str, Any], output_dir: str = "fraud_reports") -> Dict[str, str]:
        """
        Export fraud analysis to structured CSV files.
        
        Args:
            results: Analysis results from fraud detection
            output_dir: Directory to save CSV files
            
        Returns:
            Dictionary mapping severity level to CSV file path
        """
        if 'error' in results:
            self.logger.error(f"Cannot export CSV - analysis error: {results['error']}")
            return {}
        
        anomalies = results.get('anomalies', [])
        if not anomalies:
            self.logger.info("No anomalies to export")
            return {}
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Categorize anomalies by fraud severity
        fraud_categories = {
            'confirmed_fraud': [a for a in anomalies if a['price_deviation_pct'] > 1000],  # >10x price
            'likely_fraud': [a for a in anomalies if 500 < a['price_deviation_pct'] <= 1000],  # 5-10x price
            'suspicious_activity': [a for a in anomalies if 100 < a['price_deviation_pct'] <= 500],  # 2-5x price
            'moderate_overpricing': [a for a in anomalies if 30 < a['price_deviation_pct'] <= 100]  # 30-100% higher
        }
        
        # Generate timestamp for file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}
        
        for category, items in fraud_categories.items():
            if not items:
                continue
                
            filename = f"{category}_{timestamp}.csv"
            filepath = output_path / filename
            
            # Export to CSV
            self._write_fraud_csv(items, filepath, category)
            exported_files[category] = str(filepath)
            
            self.logger.info(f"Exported {len(items)} {category.replace('_', ' ')} items to {filepath}")
        
        # Create summary CSV with all anomalies
        if anomalies:
            summary_filename = f"fraud_summary_{timestamp}.csv"
            summary_filepath = output_path / summary_filename
            self._write_fraud_csv(anomalies, summary_filepath, "all_anomalies")
            exported_files['summary'] = str(summary_filepath)
            
            self.logger.info(f"Exported summary of {len(anomalies)} total anomalies to {summary_filepath}")
        
        return exported_files
    
    def _write_fraud_csv(self, anomalies: List[Dict[str, Any]], filepath: Path, category: str):
        """Write fraud data to CSV file with proper headers and formatting."""
        
        # Define CSV headers
        headers = [
            'Item_Code',
            'Item_Name', 
            'City',
            'Current_Price',
            'Normal_Price',
            'Price_Multiplier',
            'Price_Deviation_Percent',
            'Fraud_Confidence',
            'Fraud_Level',
            'Z_Score',
            'Evidence',
            'Timestamp',
            'Baseline_Data_Points'
        ]
        
        # Map category to fraud level
        fraud_levels = {
            'confirmed_fraud': 'CONFIRMED_FRAUD',
            'likely_fraud': 'LIKELY_FRAUD', 
            'suspicious_activity': 'SUSPICIOUS',
            'moderate_overpricing': 'MODERATE',
            'all_anomalies': 'MIXED'
        }
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for anomaly in anomalies:
                item_code = anomaly['item']
                item_name = self.translator.translate_item_name(item_code)
                city = anomaly['city']
                current_price = anomaly['current_price']
                normal_price = anomaly['baseline_median']
                multiplier = current_price / normal_price if normal_price > 0 else 0
                deviation_pct = anomaly['price_deviation_pct']
                confidence = anomaly['confidence']
                z_score = anomaly['z_score']
                evidence = '; '.join(anomaly['anomaly_reasons'])
                timestamp = anomaly['timestamp']
                data_points = anomaly['baseline_data_points']
                
                # Determine fraud level based on deviation
                if deviation_pct > 1000:
                    fraud_level = 'CONFIRMED_FRAUD'
                elif deviation_pct > 500:
                    fraud_level = 'LIKELY_FRAUD'
                elif deviation_pct > 100:
                    fraud_level = 'SUSPICIOUS'
                else:
                    fraud_level = 'MODERATE'
                
                writer.writerow([
                    item_code,
                    item_name,
                    city,
                    f"{current_price:.0f}",
                    f"{normal_price:.0f}",
                    f"{multiplier:.2f}",
                    f"{deviation_pct:.1f}",
                    f"{confidence:.3f}",
                    fraud_level,
                    f"{z_score:.2f}",
                    evidence,
                    timestamp,
                    data_points
                ])


def export_to_csv(results: Dict[str, Any], output_dir: str = "fraud_reports") -> Dict[str, str]:
    """
    Main function to export fraud analysis results to CSV files.
    
    Args:
        results: Analysis results from market manipulation detection
        output_dir: Directory to save CSV files
        
    Returns:
        Dictionary mapping fraud category to CSV file path
    """
    exporter = CSVFraudReporter()
    return exporter.export_fraud_analysis_csv(results, output_dir)
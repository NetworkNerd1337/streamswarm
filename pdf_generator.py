"""
PDF Report Generator for StreamSwarm Test Results
"""
import io
import os
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgets.markers import makeMarker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tempfile
import base64
from models import Test, TestResult, Client
from app import db


class StreamSwarmPDFReport:
    def __init__(self, test_id):
        self.test_id = test_id
        self.test = Test.query.get(test_id)
        if not self.test:
            raise ValueError(f"Test {test_id} not found")
        
        self.results = TestResult.query.filter_by(test_id=test_id).order_by(TestResult.timestamp).all()
        self.clients = Client.query.join(TestResult).filter(TestResult.test_id == test_id).distinct().all()
    
    def _safe_avg(self, values):
        """Helper function to safely calculate averages"""
        filtered = [v for v in values if v is not None and v != 0]
        return sum(filtered) / len(filtered) if filtered else None
    
    def _safe_min_max(self, values):
        """Helper function to safely get min/max"""
        filtered = [v for v in values if v is not None]
        if not filtered:
            return None, None
        return min(filtered), max(filtered)
        
    def generate_report(self, output_path=None):
        """Generate executive PDF report"""
        if not output_path:
            output_path = f"test_{self.test_id}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#34495e')
        )
        
        # Header with logo (if available)
        try:
            logo_path = 'static/images/logo.png'
            if os.path.exists(logo_path):
                logo = Image(logo_path, width=2*inch, height=0.8*inch)
                story.append(logo)
                story.append(Spacer(1, 12))
        except:
            pass
        
        # Title
        story.append(Paragraph("Network Performance Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", subtitle_style))
        summary_data = self._generate_executive_summary()
        story.append(Paragraph(summary_data, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Test Overview
        story.append(Paragraph("Test Configuration", subtitle_style))
        test_overview = self._generate_test_overview()
        story.append(test_overview)
        story.append(Spacer(1, 20))
        
        # Key Metrics Summary
        story.append(Paragraph("Key Performance Indicators", subtitle_style))
        metrics_tables = self._generate_metrics_table()
        if metrics_tables:
            story.extend(metrics_tables)
        story.append(Spacer(1, 20))
        
        # Performance Charts
        chart_images = self._generate_performance_charts()
        for chart_title, chart_image in chart_images:
            story.append(Paragraph(chart_title, subtitle_style))
            story.append(chart_image)
            story.append(Spacer(1, 20))
        
        # Client Performance Analysis
        story.append(PageBreak())
        story.append(Paragraph("Client Performance Analysis", subtitle_style))
        client_analysis = self._generate_client_analysis()
        if client_analysis:
            story.extend(client_analysis)
        story.append(Spacer(1, 20))
        
        # QoS Analysis (if available)
        if any(r.dscp_value is not None for r in self.results):
            story.append(Paragraph("Quality of Service Analysis", subtitle_style))
            qos_analysis = self._generate_qos_analysis()
            if qos_analysis:
                story.extend(qos_analysis)
            story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", subtitle_style))
        recommendations = self._generate_recommendations()
        story.append(Paragraph(recommendations, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 40))
        footer_text = f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>StreamSwarm Network Monitoring System"
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, textColor=colors.grey)
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        return output_path
    
    def _generate_executive_summary(self):
        """Generate executive summary text"""
        total_measurements = len(self.results)
        test_duration = (self.test.completed_at - self.test.started_at).total_seconds() / 3600 if self.test.completed_at else 0
        
        latency_values = [r.ping_latency for r in self.results if r.ping_latency is not None]
        packet_loss_values = [r.ping_packet_loss for r in self.results if r.ping_packet_loss is not None]
        
        avg_latency = self._safe_avg(latency_values) or 0
        avg_packet_loss = self._safe_avg(packet_loss_values) or 0
        
        summary = f"""
        This report analyzes network performance data collected from {len(self.clients)} monitoring locations 
        testing connectivity to {self.test.destination} over {test_duration:.1f} hours. 
        A total of {total_measurements} measurements were collected during the test period.
        
        <b>Key Findings:</b><br/>
        • Average network latency: {avg_latency:.1f} ms<br/>
        • Average packet loss: {avg_packet_loss:.2f}%<br/>
        • Test completed: {self.test.status}<br/>
        • Monitoring clients: {len(self.clients)} locations
        """
        return summary
    
    def _generate_test_overview(self):
        """Generate test configuration table"""
        data = [
            ['Test Name', self.test.name],
            ['Destination', self.test.destination],
            ['Start Time', self.test.started_at.strftime('%Y-%m-%d %H:%M:%S') if self.test.started_at else 'N/A'],
            ['End Time', self.test.completed_at.strftime('%Y-%m-%d %H:%M:%S') if self.test.completed_at else 'N/A'],
            ['Duration', f"{self.test.duration} seconds"],
            ['Measurement Interval', f"{self.test.interval} seconds"],
            ['Status', self.test.status.upper()],
            ['Participating Clients', str(len(self.clients))]
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return [table]
    
    def _generate_metrics_table(self):
        """Generate comprehensive metrics summary tables"""
        if not self.results:
            return []
        

        
        tables = []
        
        # Network Performance Metrics Table
        network_data = [
            ['Network Performance Metrics', 'Average', 'Min', 'Max', 'Status'],
        ]
        
        latencies = [r.ping_latency for r in self.results if r.ping_latency is not None]
        if latencies:
            avg_lat = self._safe_avg(latencies)
            min_lat, max_lat = self._safe_min_max(latencies)
            if avg_lat is not None and min_lat is not None and max_lat is not None:
                status = 'Excellent' if avg_lat < 50 else 'Good' if avg_lat < 100 else 'Poor'
                network_data.append(['Ping Latency (ms)', f'{avg_lat:.1f}', f'{min_lat:.1f}', f'{max_lat:.1f}', status])
        
        packet_losses = [r.ping_packet_loss for r in self.results if r.ping_packet_loss is not None]
        if packet_losses:
            avg_loss = self._safe_avg(packet_losses)
            min_loss, max_loss = self._safe_min_max(packet_losses)
            if avg_loss is not None and min_loss is not None and max_loss is not None:
                status = 'Excellent' if avg_loss < 1 else 'Good' if avg_loss < 3 else 'Poor'
                network_data.append(['Packet Loss (%)', f'{avg_loss:.2f}', f'{min_loss:.2f}', f'{max_loss:.2f}', status])
        
        jitters = [r.jitter for r in self.results if r.jitter is not None]
        if jitters:
            avg_jitter = self._safe_avg(jitters)
            min_jitter, max_jitter = self._safe_min_max(jitters)
            if avg_jitter is not None and min_jitter is not None and max_jitter is not None:
                status = 'Excellent' if avg_jitter < 10 else 'Good' if avg_jitter < 30 else 'Poor'
                network_data.append(['Network Jitter (ms)', f'{avg_jitter:.2f}', f'{min_jitter:.2f}', f'{max_jitter:.2f}', status])
        
        dns_times = [r.dns_resolution_time for r in self.results if r.dns_resolution_time is not None]
        if dns_times:
            avg_dns = self._safe_avg(dns_times)
            min_dns, max_dns = self._safe_min_max(dns_times)
            if avg_dns is not None and min_dns is not None and max_dns is not None:
                status = 'Excellent' if avg_dns < 20 else 'Good' if avg_dns < 50 else 'Poor'
                network_data.append(['DNS Resolution (ms)', f'{avg_dns:.1f}', f'{min_dns:.1f}', f'{max_dns:.1f}', status])
        
        bandwidths_down = [r.bandwidth_download for r in self.results if r.bandwidth_download is not None]
        if bandwidths_down:
            avg_down = self._safe_avg(bandwidths_down)
            min_down, max_down = self._safe_min_max(bandwidths_down)
            if avg_down is not None and min_down is not None and max_down is not None:
                status = 'Excellent' if avg_down > 100 else 'Good' if avg_down > 25 else 'Poor'
                network_data.append(['Download Speed (Mbps)', f'{avg_down:.1f}', f'{min_down:.1f}', f'{max_down:.1f}', status])
        
        if len(network_data) > 1:
            network_table = Table(network_data, colWidths=[2*inch, 1*inch, 0.8*inch, 0.8*inch, 1*inch])
            network_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            tables.extend([network_table, Spacer(1, 8)])
        
        # System Resources Table
        system_data = [
            ['System Resource Metrics', 'Average', 'Min', 'Max', 'Status'],
        ]
        
        cpu_percents = [r.cpu_percent for r in self.results if r.cpu_percent is not None]
        if cpu_percents:
            avg_cpu = self._safe_avg(cpu_percents)
            min_cpu, max_cpu = self._safe_min_max(cpu_percents)
            if avg_cpu is not None and min_cpu is not None and max_cpu is not None:
                status = 'Excellent' if avg_cpu < 50 else 'Good' if avg_cpu < 80 else 'Poor'
                system_data.append(['CPU Usage (%)', f'{avg_cpu:.1f}', f'{min_cpu:.1f}', f'{max_cpu:.1f}', status])
        
        memory_percents = [r.memory_percent for r in self.results if r.memory_percent is not None]
        if memory_percents:
            avg_mem = self._safe_avg(memory_percents)
            min_mem, max_mem = self._safe_min_max(memory_percents)
            if avg_mem is not None and min_mem is not None and max_mem is not None:
                status = 'Excellent' if avg_mem < 60 else 'Good' if avg_mem < 80 else 'Poor'
                system_data.append(['Memory Usage (%)', f'{avg_mem:.1f}', f'{min_mem:.1f}', f'{max_mem:.1f}', status])
        
        disk_percents = [r.disk_percent for r in self.results if r.disk_percent is not None]
        if disk_percents:
            avg_disk = self._safe_avg(disk_percents)
            min_disk, max_disk = self._safe_min_max(disk_percents)
            if avg_disk is not None and min_disk is not None and max_disk is not None:
                status = 'Excellent' if avg_disk < 70 else 'Good' if avg_disk < 85 else 'Poor'
                system_data.append(['Disk Usage (%)', f'{avg_disk:.1f}', f'{min_disk:.1f}', f'{max_disk:.1f}', status])
        
        load_1mins = [r.cpu_load_1min for r in self.results if r.cpu_load_1min is not None]
        if load_1mins:
            avg_load = self._safe_avg(load_1mins)
            min_load, max_load = self._safe_min_max(load_1mins)
            if avg_load is not None and min_load is not None and max_load is not None:
                status = 'Excellent' if avg_load < 1 else 'Good' if avg_load < 2 else 'Poor'
                system_data.append(['CPU Load (1min)', f'{avg_load:.2f}', f'{min_load:.2f}', f'{max_load:.2f}', status])
        
        if len(system_data) > 1:
            system_table = Table(system_data, colWidths=[2*inch, 1*inch, 0.8*inch, 0.8*inch, 1*inch])
            system_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            tables.extend([system_table, Spacer(1, 8)])
        
        # Advanced Network & QoS Table
        advanced_data = [
            ['Advanced Network & QoS', 'Value', 'Analysis', '', ''],
        ]
        
        mtu_sizes = [r.mtu_size for r in self.results if r.mtu_size is not None]
        if mtu_sizes:
            avg_mtu = self._safe_avg(mtu_sizes)
            advanced_data.append(['MTU Size (bytes)', f'{avg_mtu:.0f}', 'Standard', '', ''])
        
        retrans_rates = [r.tcp_retransmission_rate for r in self.results if r.tcp_retransmission_rate is not None]
        if retrans_rates:
            avg_retrans = self._safe_avg(retrans_rates)
            status = 'Excellent' if avg_retrans < 0.5 else 'Good' if avg_retrans < 2 else 'Poor'
            advanced_data.append(['TCP Retrans Rate (%)', f'{avg_retrans:.3f}', status, '', ''])
        
        ecn_capable = any(r.ecn_capable for r in self.results if r.ecn_capable)
        advanced_data.append(['ECN Support', 'Yes' if ecn_capable else 'No', 'Info', '', ''])
        
        policing_detected = any(r.traffic_policing_detected for r in self.results if r.traffic_policing_detected)
        advanced_data.append(['Traffic Policing', 'Detected' if policing_detected else 'None', 'Info', '', ''])
        
        if len(advanced_data) > 1:
            advanced_table = Table(advanced_data, colWidths=[2*inch, 1*inch, 1*inch, 0.8*inch, 0.8*inch])
            advanced_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f39c12')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            tables.extend([advanced_table, Spacer(1, 8)])
        
        # Application Layer & Infrastructure Table
        app_data = [
            ['Application & Infrastructure', 'Value', 'Performance', '', ''],
        ]
        
        content_times = [r.content_download_time for r in self.results if r.content_download_time is not None]
        if content_times:
            avg_content = self._safe_avg(content_times)
            status = 'Excellent' if avg_content < 1000 else 'Good' if avg_content < 3000 else 'Poor'
            app_data.append(['Content Download (ms)', f'{avg_content:.0f}', status, '', ''])
        
        compression_ratios = [r.compression_ratio for r in self.results if r.compression_ratio is not None]
        if compression_ratios:
            avg_compression = self._safe_avg(compression_ratios)
            status = 'Excellent' if avg_compression > 60 else 'Good' if avg_compression > 30 else 'Poor'
            app_data.append(['Compression Ratio (%)', f'{avg_compression:.1f}', status, '', ''])
        
        power_consumption = [r.power_consumption_watts for r in self.results if r.power_consumption_watts is not None]
        if power_consumption:
            avg_power = self._safe_avg(power_consumption)
            app_data.append(['Power Consumption (W)', f'{avg_power:.1f}', 'Normal', '', ''])
        
        memory_errors = [r.memory_error_rate for r in self.results if r.memory_error_rate is not None]
        if memory_errors:
            avg_errors = self._safe_avg(memory_errors)
            status = 'Excellent' if avg_errors < 0.1 else 'Good' if avg_errors < 1 else 'Monitor'
            app_data.append(['Memory Errors (/hr)', f'{avg_errors:.3f}', status, '', ''])
        
        if len(app_data) > 1:
            app_table = Table(app_data, colWidths=[2*inch, 1*inch, 1*inch, 0.8*inch, 0.8*inch])
            app_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            tables.extend([app_table, Spacer(1, 8)])
        
        return tables
    
    def _generate_performance_charts(self):
        """Generate performance charts as images"""
        charts = []
        
        # Latency over time chart
        if any(r.ping_latency for r in self.results):
            latency_chart = self._create_latency_chart()
            charts.append(("Network Latency Over Time", latency_chart))
        
        # Client comparison chart
        if len(self.clients) > 1:
            comparison_chart = self._create_client_comparison_chart()
            charts.append(("Client Performance Comparison", comparison_chart))
        
        return charts
    
    def _create_latency_chart(self):
        """Create latency over time chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for client in self.clients:
            client_results = [r for r in self.results if r.client_id == client.id and r.ping_latency]
            if client_results:
                timestamps = [r.timestamp for r in client_results]
                latencies = [r.ping_latency for r in client_results]
                ax.plot(timestamps, latencies, label=client.hostname, marker='o', markersize=3)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Network Latency Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save to temporary file and convert to ReportLab image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
        plt.close()
        
        image = Image(temp_file.name, width=7*inch, height=4*inch)
        os.unlink(temp_file.name)  # Clean up temp file
        
        return image
    
    def _create_client_comparison_chart(self):
        """Create client performance comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        client_names = []
        avg_latencies = []
        
        for client in self.clients:
            client_results = [r for r in self.results if r.client_id == client.id and r.ping_latency]
            if client_results:
                avg_latency = sum(r.ping_latency for r in client_results) / len(client_results)
                client_names.append(client.hostname)
                avg_latencies.append(avg_latency)
        
        bars = ax.bar(client_names, avg_latencies, color='skyblue', edgecolor='navy')
        ax.set_ylabel('Average Latency (ms)')
        ax.set_title('Average Network Latency by Client Location')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}ms',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to temporary file and convert to ReportLab image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
        plt.close()
        
        image = Image(temp_file.name, width=7*inch, height=4*inch)
        os.unlink(temp_file.name)  # Clean up temp file
        
        return image
    
    def _generate_client_analysis(self):
        """Generate detailed client analysis table"""
        data = [['Client Location', 'Avg Latency (ms)', 'Avg Packet Loss (%)', 'Measurements', 'Status']]
        
        for client in self.clients:
            client_results = [r for r in self.results if r.client_id == client.id]
            
            latencies = [r.ping_latency for r in client_results if r.ping_latency is not None]
            packet_losses = [r.ping_packet_loss for r in client_results if r.ping_packet_loss is not None]
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            avg_packet_loss = sum(packet_losses) / len(packet_losses) if packet_losses else 0
            
            # Determine status
            if avg_latency < 50 and avg_packet_loss < 1:
                status = "Excellent"
            elif avg_latency < 100 and avg_packet_loss < 3:
                status = "Good"
            else:
                status = "Needs Attention"
            
            data.append([
                client.hostname,
                f"{avg_latency:.1f}" if latencies else "N/A",
                f"{avg_packet_loss:.2f}" if packet_losses else "N/A",
                str(len(client_results)),
                status
            ])
        
        table = Table(data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return [table]
    
    def _generate_qos_analysis(self):
        """Generate QoS analysis if QoS data is available"""
        qos_results = [r for r in self.results if r.dscp_value is not None]
        
        if not qos_results:
            return [Paragraph("No QoS data available for this test.", getSampleStyleSheet()['Normal'])]
        
        # Analyze DSCP distribution
        dscp_counts = {}
        for result in qos_results:
            dscp = result.dscp_value
            if dscp in dscp_counts:
                dscp_counts[dscp] += 1
            else:
                dscp_counts[dscp] = 1
        
        analysis_text = "<b>Quality of Service Analysis:</b><br/><br/>"
        analysis_text += f"QoS data collected from {len(qos_results)} measurements.<br/><br/>"
        analysis_text += "<b>DSCP Value Distribution:</b><br/>"
        
        for dscp, count in sorted(dscp_counts.items()):
            percentage = (count / len(qos_results)) * 100
            traffic_class = self._classify_dscp(dscp)
            analysis_text += f"• DSCP {dscp} ({traffic_class}): {count} measurements ({percentage:.1f}%)<br/>"
        
        return [Paragraph(analysis_text, getSampleStyleSheet()['Normal'])]
    
    def _classify_dscp(self, dscp_value):
        """Classify traffic based on DSCP value"""
        dscp_map = {
            0: "Best Effort",
            10: "AF11", 12: "AF12", 14: "AF13",
            18: "AF21", 20: "AF22", 22: "AF23",
            26: "AF31", 28: "AF32", 30: "AF33",
            34: "AF41", 36: "AF42", 38: "AF43",
            46: "Expedited Forwarding",
            8: "CS1", 16: "CS2", 24: "CS3", 32: "CS4", 40: "CS5", 48: "CS6", 56: "CS7"
        }
        return dscp_map.get(dscp_value, f"Unknown ({dscp_value})")
    
    def _generate_recommendations(self):
        """Generate comprehensive actionable recommendations based on all 65+ metrics"""
        if not self.results:
            return ""
        
        def safe_avg(values):
            filtered = [v for v in values if v is not None and v != 0]
            return sum(filtered) / len(filtered) if filtered else None
        
        recommendations = []
        
        # Network Performance Analysis
        latencies = [r.ping_latency for r in self.results if r.ping_latency is not None]
        if latencies:
            avg_latency = self._safe_avg(latencies)
            if avg_latency and avg_latency > 150:
                recommendations.append("CRITICAL: High network latency >150ms detected. Implement CDN, optimize routing, or consider edge computing solutions.")
            elif avg_latency and avg_latency > 100:
                recommendations.append("HIGH: Elevated latency 100-150ms. Monitor consistency and investigate network path optimization.")
        
        packet_losses = [r.ping_packet_loss for r in self.results if r.ping_packet_loss is not None]
        if packet_losses:
            avg_loss = self._safe_avg(packet_losses)
            if avg_loss and avg_loss > 5:
                recommendations.append("CRITICAL: Packet loss >5% severely impacts performance. Immediate network infrastructure review required.")
            elif avg_loss and avg_loss > 1:
                recommendations.append("MEDIUM: Packet loss 1-5% detected. Monitor during peak usage and review QoS policies.")
        
        jitters = [r.jitter for r in self.results if r.jitter is not None]
        if jitters:
            avg_jitter = self._safe_avg(jitters)
            if avg_jitter and avg_jitter > 30:
                recommendations.append("HIGH: High jitter >30ms impacts real-time applications. Investigate QoS policies and network stability.")
        
        # System Resource Analysis
        cpu_percents = [r.cpu_percent for r in self.results if r.cpu_percent is not None]
        if cpu_percents:
            avg_cpu = self._safe_avg(cpu_percents)
            if avg_cpu and avg_cpu > 90:
                recommendations.append("CRITICAL: CPU >90% utilization. Immediate scaling or workload optimization required.")
            elif avg_cpu and avg_cpu > 80:
                recommendations.append("HIGH: CPU 80-90% usage. Plan capacity expansion and optimize resource-intensive processes.")
        
        memory_percents = [r.memory_percent for r in self.results if r.memory_percent is not None]
        if memory_percents:
            avg_memory = self._safe_avg(memory_percents)
            if avg_memory and avg_memory > 95:
                recommendations.append("CRITICAL: Memory >95% usage. Immediate memory optimization or hardware upgrade required.")
            elif avg_memory and avg_memory > 85:
                recommendations.append("HIGH: Memory 85-95% usage. Monitor for memory leaks and plan capacity increases.")
        
        # Advanced Network Analysis
        retrans_rates = [r.tcp_retransmission_rate for r in self.results if r.tcp_retransmission_rate is not None]
        if retrans_rates:
            avg_retrans = self._safe_avg(retrans_rates)
            if avg_retrans and avg_retrans > 2:
                recommendations.append("MEDIUM: TCP retransmission rate >2%. Investigate network congestion and buffer sizing.")
        
        # QoS Analysis
        ecn_capable = any(r.ecn_capable for r in self.results if r.ecn_capable)
        if not ecn_capable:
            recommendations.append("INFO: ECN not detected. Consider enabling ECN for improved congestion handling.")
        
        policing_detected = any(r.traffic_policing_detected for r in self.results if r.traffic_policing_detected)
        if policing_detected:
            recommendations.append("INFO: Traffic policing detected. Verify QoS policies align with application requirements.")
        
        # Application Layer Analysis
        content_times = [r.content_download_time for r in self.results if r.content_download_time is not None]
        if content_times:
            avg_content = self._safe_avg(content_times)
            if avg_content and avg_content > 5000:  # 5 seconds
                recommendations.append("MEDIUM: Slow content download >5s. Optimize content delivery, enable compression, or implement CDN.")
        
        compression_ratios = [r.compression_ratio for r in self.results if r.compression_ratio is not None]
        if compression_ratios:
            avg_compression = self._safe_avg(compression_ratios)
            if avg_compression and avg_compression < 20:
                recommendations.append("LOW: Low compression ratio <20%. Enable gzip/deflate compression to improve transfer efficiency.")
        
        # Infrastructure Health Analysis
        memory_errors = [r.memory_error_rate for r in self.results if r.memory_error_rate is not None]
        if memory_errors:
            avg_errors = self._safe_avg(memory_errors)
            if avg_errors and avg_errors > 1:
                recommendations.append("HIGH: Memory errors detected >1/hr. Investigate RAM health and consider ECC memory.")
        
        power_consumption = [r.power_consumption_watts for r in self.results if r.power_consumption_watts is not None]
        if power_consumption:
            avg_power = self._safe_avg(power_consumption)
            if avg_power and avg_power > 200:
                recommendations.append("INFO: High power consumption >200W. Consider energy optimization for cost reduction.")
        
        # Bandwidth Analysis
        bandwidths_down = [r.bandwidth_download for r in self.results if r.bandwidth_download is not None]
        if bandwidths_down:
            avg_down = self._safe_avg(bandwidths_down)
            if avg_down and avg_down < 10:
                recommendations.append("MEDIUM: Low download bandwidth <10 Mbps. Consider bandwidth upgrade for improved user experience.")
        
        # Client Performance Comparison
        if len(self.clients) > 1:
            client_performance = {}
            for client in self.clients:
                client_results = [r for r in self.results if r.client_id == client.id]
                if client_results:
                    client_latencies = [r.ping_latency for r in client_results if r.ping_latency is not None]
                    if client_latencies:
                        avg_latency = self._safe_avg(client_latencies)
                        if avg_latency is not None:
                            client_performance[client.hostname] = avg_latency
            
            if len(client_performance) > 1:
                # Filter out None values before comparison
                valid_performance = {k: v for k, v in client_performance.items() if v is not None}
                if len(valid_performance) > 1:
                    best_client = min(valid_performance.items(), key=lambda x: x[1])
                    worst_client = max(valid_performance.items(), key=lambda x: x[1])
                    
                    if worst_client[1] > best_client[1] * 1.5:
                        recommendations.append(f"MEDIUM: Performance disparity detected. {worst_client[0]} shows significantly higher latency than {best_client[0]}. Investigate network path optimization.")
        
        # General recommendations if performance is good
        if not recommendations:
            recommendations.extend([
                "EXCELLENT: All metrics within optimal ranges. System performing well.",
                "Continue regular monitoring to maintain performance baselines.",
                "Consider implementing automated alerting for proactive issue detection."
            ])
        else:
            recommendations.extend([
                "Implement continuous monitoring for early issue detection.",
                "Schedule regular performance reviews to track improvements.",
                "Consider capacity planning based on growth projections."
            ])
        
        return "<br/>".join(f"• {rec}" for rec in recommendations)


def generate_test_report_pdf(test_id, output_path=None):
    """Generate PDF report for a specific test"""
    try:
        report_generator = StreamSwarmPDFReport(test_id)
        return report_generator.generate_report(output_path)
    except Exception as e:
        raise Exception(f"Failed to generate PDF report: {str(e)}")
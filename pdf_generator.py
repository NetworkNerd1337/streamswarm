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
        metrics_table = self._generate_metrics_table()
        story.append(metrics_table)
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
        story.append(client_analysis)
        story.append(Spacer(1, 20))
        
        # QoS Analysis (if available)
        if any(r.dscp_value is not None for r in self.results):
            story.append(Paragraph("Quality of Service Analysis", subtitle_style))
            qos_analysis = self._generate_qos_analysis()
            story.append(qos_analysis)
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
        
        avg_latency = sum(r.ping_latency for r in self.results if r.ping_latency) / max(1, len([r for r in self.results if r.ping_latency]))
        avg_packet_loss = sum(r.ping_packet_loss for r in self.results if r.ping_packet_loss) / max(1, len([r for r in self.results if r.ping_packet_loss]))
        
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
        
        return table
    
    def _generate_metrics_table(self):
        """Generate key metrics summary table"""
        # Calculate aggregated metrics
        latencies = [r.ping_latency for r in self.results if r.ping_latency is not None]
        packet_losses = [r.ping_packet_loss for r in self.results if r.ping_packet_loss is not None]
        dns_times = [r.dns_resolution_time for r in self.results if r.dns_resolution_time is not None]
        tcp_times = [r.tcp_connect_time for r in self.results if r.tcp_connect_time is not None]
        
        data = [
            ['Metric', 'Average', 'Minimum', 'Maximum', 'Status'],
            ['Network Latency (ms)', 
             f"{sum(latencies)/max(1,len(latencies)):.1f}" if latencies else 'N/A',
             f"{min(latencies):.1f}" if latencies else 'N/A',
             f"{max(latencies):.1f}" if latencies else 'N/A',
             'Excellent' if latencies and sum(latencies)/len(latencies) < 50 else 'Good' if latencies and sum(latencies)/len(latencies) < 100 else 'Poor'
            ],
            ['Packet Loss (%)',
             f"{sum(packet_losses)/max(1,len(packet_losses)):.2f}" if packet_losses else 'N/A',
             f"{min(packet_losses):.2f}" if packet_losses else 'N/A',
             f"{max(packet_losses):.2f}" if packet_losses else 'N/A',
             'Excellent' if packet_losses and sum(packet_losses)/len(packet_losses) < 1 else 'Good' if packet_losses and sum(packet_losses)/len(packet_losses) < 3 else 'Poor'
            ],
            ['DNS Resolution (ms)',
             f"{sum(dns_times)/max(1,len(dns_times)):.1f}" if dns_times else 'N/A',
             f"{min(dns_times):.1f}" if dns_times else 'N/A',
             f"{max(dns_times):.1f}" if dns_times else 'N/A',
             'Excellent' if dns_times and sum(dns_times)/len(dns_times) < 20 else 'Good' if dns_times and sum(dns_times)/len(dns_times) < 50 else 'Poor'
            ],
            ['TCP Connect (ms)',
             f"{sum(tcp_times)/max(1,len(tcp_times)):.1f}" if tcp_times else 'N/A',
             f"{min(tcp_times):.1f}" if tcp_times else 'N/A',
             f"{max(tcp_times):.1f}" if tcp_times else 'N/A',
             'Excellent' if tcp_times and sum(tcp_times)/len(tcp_times) < 100 else 'Good' if tcp_times and sum(tcp_times)/len(tcp_times) < 200 else 'Poor'
            ]
        ]
        
        table = Table(data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            # Color code status column
            ('TEXTCOLOR', (4, 1), (4, -1), colors.green),  # Will be overridden by actual status colors
        ]))
        
        return table
    
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
        
        return table
    
    def _generate_qos_analysis(self):
        """Generate QoS analysis if QoS data is available"""
        qos_results = [r for r in self.results if r.dscp_value is not None]
        
        if not qos_results:
            return Paragraph("No QoS data available for this test.", getSampleStyleSheet()['Normal'])
        
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
        
        return Paragraph(analysis_text, getSampleStyleSheet()['Normal'])
    
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
        """Generate actionable recommendations"""
        # Analyze results to provide recommendations
        latencies = [r.ping_latency for r in self.results if r.ping_latency is not None]
        packet_losses = [r.ping_packet_loss for r in self.results if r.ping_packet_loss is not None]
        
        recommendations = []
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            if avg_latency > 100:
                recommendations.append("High network latency detected. Consider optimizing network routing or upgrading connection quality.")
        
        if packet_losses:
            avg_packet_loss = sum(packet_losses) / len(packet_losses)
            if avg_packet_loss > 3:
                recommendations.append("Significant packet loss observed. Investigate network infrastructure for capacity or reliability issues.")
        
        if len(self.clients) > 1:
            # Compare client performance
            client_latencies = {}
            for client in self.clients:
                client_results = [r for r in self.results if r.client_id == client.id and r.ping_latency]
                if client_results:
                    client_latencies[client.hostname] = sum(r.ping_latency for r in client_results) / len(client_results)
            
            if len(client_latencies) > 1:
                best_client = min(client_latencies.items(), key=lambda x: x[1])
                worst_client = max(client_latencies.items(), key=lambda x: x[1])
                
                if worst_client[1] > best_client[1] * 1.5:
                    recommendations.append(f"Performance disparity detected: {worst_client[0]} shows significantly higher latency than {best_client[0]}. Consider investigating network path optimization.")
        
        if not recommendations:
            recommendations.append("Network performance appears to be within acceptable parameters. Continue regular monitoring to maintain service quality.")
        
        return "<br/>".join(f"• {rec}" for rec in recommendations)


def generate_test_report_pdf(test_id, output_path=None):
    """Generate PDF report for a specific test"""
    try:
        report_generator = StreamSwarmPDFReport(test_id)
        return report_generator.generate_report(output_path)
    except Exception as e:
        raise Exception(f"Failed to generate PDF report: {str(e)}")
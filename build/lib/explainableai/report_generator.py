# explainableai/report_generation.py

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage

class ReportGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        self.styles = getSampleStyleSheet()
        self.styles['Heading2'].fontSize = 14
        self.styles['Heading2'].spaceBefore = 12
        self.styles['Heading2'].spaceAfter = 6
        self.content = []

    def add_heading(self, text, level=1):
        style = self.styles['Heading1'] if level == 1 else self.styles['Heading2']
        self.content.append(Paragraph(text, style))
        self.content.append(Spacer(1, 12))

    def add_paragraph(self, text):
        self.content.append(Paragraph(text, self.styles['Normal']))
        self.content.append(Spacer(1, 6))

    def add_image(self, image_path, width=6*inch, height=4*inch):
        try:
            img = PILImage.open(image_path)
            img.thumbnail((width, height), PILImage.LANCZOS)
            img_data = io.BytesIO()
            img.save(img_data, format='PNG')
            img_data.seek(0)
            img = Image(img_data, width=width, height=height)
            self.content.append(img)
            self.content.append(Spacer(1, 12))
        except Exception as e:
            print(f"Error adding image {image_path}: {str(e)}")

    def add_table(self, data, col_widths=None):
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.content.append(table)
        self.content.append(Spacer(1, 12))

    def add_plot(self, fig, width=6*inch, height=4*inch):
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=300, bbox_inches='tight')
        img_data.seek(0)
        img = Image(img_data, width=width, height=height)
        self.content.append(img)
        self.content.append(Spacer(1, 12))
        plt.close(fig)

    def generate(self):
        try:
            self.doc.build(self.content)
            print(f"Report generated successfully: {self.filename}")
        except Exception as e:
            print(f"Error generating report: {str(e)}")
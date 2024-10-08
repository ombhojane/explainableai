# explainableai/report_generation.py

import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from PIL import Image as PILImage
import re
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

class ReportGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        self.styles = getSampleStyleSheet()
        self.custom_styles = {}
        self.setup_styles()
        self.content = []

    def setup_styles(self):
        self.custom_styles['Heading1'] = ParagraphStyle(
            'CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceBefore=12,
            spaceAfter=6
        )
        
        self.custom_styles['Heading2'] = ParagraphStyle(
            'CustomHeading2',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=12,
            spaceAfter=6
        )

        self.custom_styles['Heading3'] = ParagraphStyle(
            'CustomHeading3',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=10,
            spaceAfter=4
        )

        self.custom_styles['BodyText'] = ParagraphStyle(
            'CustomBodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6
        )

    def add_heading(self, text, level=1):
        style = self.custom_styles[f'Heading{level}']
        self.content.append(Paragraph(text, style))
        self.content.append(Spacer(1, 12))

    def add_paragraph(self, text):
        formatted_text = self.format_text(text)
        self.content.append(Paragraph(formatted_text, self.custom_styles['BodyText']))
        self.content.append(Spacer(1, 6))

    def format_text(self, text):
        # Convert Markdown-style formatting to ReportLab's XML-like tags
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italic
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)  # Code
        return text

    def add_llm_explanation(self, explanation):
        lines = explanation.split('\n')
        for line in lines:
            if line.startswith('##'):
                self.add_heading(line.strip('# '), level=2)
            elif line.startswith('#'):
                self.add_heading(line.strip('# '), level=3)
            elif line.strip():
                self.add_paragraph(line)

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

    def generate(self):
        try:
            self.doc.build(self.content)
            print(f"Report generated successfully: {self.filename}")
        except Exception as e:
            print(f"Error generating report: {str(e)}")

    def create_keras_model(self, input_dim):
        # Create a Keras model for binary classification
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_keras_model(self, X, y):
        model = self.create_keras_model(X.shape[1])
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        return model

    def evaluate_keras_model(self, model, X, y):
        y_pred = (model.predict(X) > 0.5).astype(int)
        accuracy = np.mean(y_pred.flatten() == y)
        return accuracy

# Example usage:
# X_train = ...  # your training feature data as a numpy array
# y_train = ...  # your training labels as a numpy array

# report = ReportGenerator("report.pdf")
# keras_model = report.train_keras_model(X_train, y_train)  # Train the Keras model
# accuracy = report.evaluate_keras_model(keras_model, X_train, y_train)  # Evaluate the model
# report.add_heading("Keras Model Evaluation", level=1)
# report.add_paragraph(f"Model Accuracy: {accuracy:.2f}")
# report.generate()  # Generate the report
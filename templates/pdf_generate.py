from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_endometriosis_report():
    # Create a PDF file
    pdf_file = "1.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Medical Report: Endometriosis")

    # Patient Information
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 100, "Patient Information")
    c.drawString(72, height - 120, "Name: Jane Doe")
    c.drawString(72, height - 135, "Age: 28")
    c.drawString(72, height - 150, "Date: 19th October 2024")

    # Symptoms
    c.drawString(72, height - 180, "Symptoms")
    c.drawString(72, height - 200, "The patient has reported the following symptoms:")
    symptoms = ["Heavy periods"]
    for i, symptom in enumerate(symptoms):
        c.drawString(100, height - 220 - (i * 15), symptom)

    # Diagnosis
    c.drawString(72, height - 270, "Diagnosis")
    c.drawString(72, height - 285, "Based on the symptoms and clinical evaluation, the diagnosis is confirmed as Endometriosis.")

    # Treatment Recommendations
    c.drawString(72, height - 320, "Treatment Recommendations")
    recommendations = [
        "• Lifestyle changes: Incorporating a healthy diet and regular exercise.",
        "• Medications: Pain relief medications and hormonal therapies.",
        "• Surgery: In severe cases, surgical options may be considered."
    ]
    for i, recommendation in enumerate(recommendations):
        c.drawString(100, height - 340 - (i * 15), recommendation)

    # Precautions
    c.drawString(72, height - 400, "Precautions")
    precautions = [
        "• Regular monitoring of symptoms.",
        "• Maintaining a healthy weight.",
        "• Seeking support from healthcare providers."
    ]
    for i, precaution in enumerate(precautions):
        c.drawString(100, height - 420 - (i * 15), precaution)

    # Recommended Food Intake
    c.drawString(72, height - 480, "Recommended Food Intake")
    food_recommendations = [
        "• High-fiber foods such as vegetables and whole grains.",
        "• Lean protein sources like fish and poultry.",
        "• Healthy fats from nuts, seeds, and olive oil."
    ]
    for i, food in enumerate(food_recommendations):
        c.drawString(100, height - 500 - (i * 15), food)

    # Conclusion
    c.drawString(72, height - 560, "Conclusion")
    conclusion = "Endometriosis is a manageable condition with proper care and lifestyle adjustments. " \
                 "It is recommended to follow up regularly with the healthcare provider to monitor progress and make necessary treatment adjustments."
    c.drawString(72, height - 575, conclusion)

    # Save the PDF
    c.save()

    print(f"PDF report '{pdf_file}' has been created.")

# Generate the Endometriosis medical report
create_endometriosis_report()

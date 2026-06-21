import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

from simple_model import load_and_clean, preprocess, train_and_evaluate


def build_pdf(report_path, text_blocks, image_paths):
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for t in text_blocks:
        story.append(Paragraph(t, styles['Normal']))
        story.append(Spacer(1, 12))

    for img in image_paths:
        if os.path.exists(img):
            story.append(Image(img, width=400, height=250))
            story.append(Spacer(1, 12))

    doc.build(story)


def main():
    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    out_dir = os.path.join('outputs', 'report')
    os.makedirs(out_dir, exist_ok=True)

    print('Running training to collect metrics...')
    df = load_and_clean(csv_path)
    X, y = preprocess(df)
    _, results = train_and_evaluate(X, y, save_dir=out_dir)

    # Prepare text blocks
    text_blocks = [
        '<b>Telco Churn — Project Report</b>',
        f"<b>Accuracy:</b> {results['accuracy']:.4f}",
        '<b>Classification Report:</b>',
        '<pre>' + results['classification_report'].replace('\n', '<br/>') + '</pre>',
    ]

    # Collect images: EDA plots and confusion matrix
    image_paths = []
    eda_dir = os.path.join('outputs', 'eda_plots')
    if os.path.exists(eda_dir):
        for fname in sorted(os.listdir(eda_dir)):
            if fname.lower().endswith('.png'):
                image_paths.append(os.path.join(eda_dir, fname))

    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    if os.path.exists(cm_path):
        image_paths.append(cm_path)

    report_pdf = os.path.join(out_dir, 'report.pdf')
    print(f'Building PDF at {report_pdf}...')
    build_pdf(report_pdf, text_blocks, image_paths)
    print('Report generated:', report_pdf)


if __name__ == '__main__':
    main()

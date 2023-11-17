# ML-Builder

Welcome to ML-Builder, a user-friendly system that allows you to generate your own machine learning model without writing a single line of code. This project utilizes Flask for the backend, HTML, CSS, and Bootstrap 5 for the frontend. Here's a quick overview of the process:

## Project Overview

1. **Upload CSV File:**
   - Navigate to the home page and upload your CSV file.
   - The system will display all columns of your dataset.

2. **Choose Target Column:**
   - Select the target column for which you want to predict.

3. **Build Instant Model:**
   - Click on "Build Instant Model" to generate a default machine learning model using the DecisionTree algorithm.

4. **Customize Model:**
   - If you want to build a customized model, go to the Customize section.
   - Choose the algorithm you want to train.
   - Customize data processing techniques (minmax scaler, standard scaler, one hot encoding, label encoding, etc.).

5. **Get Results:**
   - The web page will return the trained model and processed datasets.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/vijaytakbhate2002/ML-Builder.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Access the application in your browser at [http://localhost:5000](http://localhost:5000).

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the [MIT License](LICENSE).

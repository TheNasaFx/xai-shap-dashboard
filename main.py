"""
XAI-SHAP Визуал Аналитик Framework
===================================

SHAP ашиглан Тайлбарлах боломжтой AI-д зориулсан иж бүрэн framework.

Хэрэглээ:
    python main.py [команд]

Командууд:
    run       - Үндсэн ашиглалтын жишээг ажиллуулах
    dashboard - Интерактив dashboard ажиллуулах
    test      - Unit тестүүд ажиллуулах
    generate  - Жишээ өгөгдлийн багцууд үүсгэх

Зохиогч: XAI-SHAP Framework
"""

import sys
import argparse


def run_example():
    """Үндсэн ашиглалтын жишээг ажиллуулах."""
    print("Үндсэн ашиглалтын жишээг ажиллуулж байна...")
    from examples.basic_usage import main
    main()


def run_dashboard():
    """Streamlit dashboard ажиллуулах."""
    print("Dashboard ажиллуулж байна...")
    print("Хөтчөөс http://localhost:8501 нээнэ үү")
    
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])


def run_tests():
    """Unit тестүүд ажиллуулах."""
    print("Тестүүд ажиллуулж байна...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def generate_data():
    """Жишээ өгөгдлийн багцууд үүсгэх."""
    print("Жишээ өгөгдлийн багцууд үүсгэж байна...")
    from data.sample.generate_samples import (
        generate_loan_dataset,
        generate_credit_risk_dataset,
        get_breast_cancer_dataset,
        generate_employee_attrition_dataset
    )
    import os
    
    os.makedirs("data/sample", exist_ok=True)
    
    generate_loan_dataset(1000, "data/sample/loan_data.csv")
    generate_credit_risk_dataset(1000, "data/sample/credit_risk.csv")
    get_breast_cancer_dataset("data/sample/breast_cancer.csv")
    generate_employee_attrition_dataset(1000, "data/sample/employee_attrition.csv")
    
    print("✅ Жишээ өгөгдлийн багцууд data/sample/ хавтаст үүсгэгдлээ")


def main():
    """Үндсэн эхлэх цэг."""
    parser = argparse.ArgumentParser(
        description="XAI-SHAP Визуал Аналитик Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Жишээнүүд:
  python main.py run          # Үндсэн жишээ ажиллуулах
  python main.py dashboard    # Dashboard ажиллуулах
  python main.py test         # Тестүүд ажиллуулах
  python main.py generate     # Жишээ өгөгдөл үүсгэх
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['run', 'dashboard', 'test', 'generate'],
        default='run',
        help='Гүйцэтгэх команд (анхдагч: run)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XAI-SHAP Визуал Аналитик Framework")
    print("Хар Хайрцагт Загваруудад зориулсан Тайлбарлах боломжтой AI")
    print("=" * 60)
    
    if args.command == 'run':
        run_example()
    elif args.command == 'dashboard':
        run_dashboard()
    elif args.command == 'test':
        run_tests()
    elif args.command == 'generate':
        generate_data()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

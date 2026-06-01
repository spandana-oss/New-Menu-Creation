from src.customer_intelligence import save_customer_intelligence
from src.merge_data import merge_datasets


def main():
    print("Building merged datasets...")
    merge_datasets()
    print("Building customer intelligence workbook...")
    save_customer_intelligence()
    print("Pipeline completed.")


if __name__ == "__main__":
    main()

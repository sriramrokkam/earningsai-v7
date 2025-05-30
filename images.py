from scripts.image_processor import process_images
# Example usage
if __name__ == "__main__":
    folder_path = "/home/user/projects/EarningsAI-Assistant-Q1-25/Images"
    user_prompt = """ Bank: BNP Paribas
Period: Q1 2025
Stock analysis and Financial Insights """
    
    responses = process_images(folder_path, user_prompt)
    
    # Optionally print returned responses
    print("\nReturned Responses:")
    for result in responses:
        print(f"Image: {result['image_path']}\nAnalysis: {result['analysis']}\n{'='*50}")
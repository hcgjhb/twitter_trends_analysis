from app import create_app
from app.services.user_clustering import cluster_users_by_tweets
import matplotlib.pyplot as plt

app = create_app()

# Create an application context
with app.app_context():
    try:
        # Run clustering with at least 2 clusters
        results = cluster_users_by_tweets(n_clusters=3, method='kmeans')
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print("Clustering completed successfully!")
            print(f"Visualization saved to: {results['visualization_path']}")
            print(f"Sentiment map saved to: {results['sentiment_visualization_path']}")
            
            # You can open the images directly if desired
            # import subprocess
            # subprocess.run(['xdg-open', results['visualization_path']])
            # subprocess.run(['xdg-open', results['sentiment_visualization_path']])
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        import traceback
        traceback.print_exc()
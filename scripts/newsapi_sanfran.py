import requests
import csv
import os
from datetime import datetime, timezone, timedelta
import time

# Define the subreddit
subreddit_name = 'sanfrancisco'

# Keywords related to homelessness
keywords = [
    'homeless', 'homelessness', 'housing crisis',
    'affordable housing', 'unhoused', 'houseless',
    'housing insecurity', 'beggar', 'squatter', 'panhandler', 'soup kitchen'
]

# Prepare the output directory
output_dir = f'data/{subreddit_name.lower()}/reddit'
os.makedirs(output_dir, exist_ok=True)
all_comments_file = os.path.join(output_dir, 'all_comments.csv')
filtered_comments_file = os.path.join(output_dir, 'filtered_comments.csv')
statistics_file = os.path.join(output_dir, 'statistics.csv')

# Function to convert UTC timestamp to a readable format with timezone support
def format_timestamp(utc_timestamp):
    dt = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')

# Function to search posts, export data, and generate statistics
def search_posts_and_export_time_chunks(subreddit_name, keywords, chunk_days=30):
    total_posts = 0
    total_comments = 0
    total_filtered_comments = 0
    total_comment_score = 0
    total_filtered_comment_score = 0

    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    time_delta = timedelta(days=chunk_days)

    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + time_delta, end_date)
        print(f"Searching from {current_start} to {current_end}")

        try:
            # Search using Pushshift API with pagination, resetting for each chunk
            after = int(current_start.timestamp())
            before = int(current_end.timestamp())
            query = ' OR '.join(keywords)
            
            # Prepare CSV files for each chunk
            chunk_all_comments_file = os.path.join(output_dir, f'all_comments_{current_start.strftime("%Y%m%d")}.csv')
            chunk_filtered_comments_file = os.path.join(output_dir, f'filtered_comments_{current_start.strftime("%Y%m%d")}.csv')

            with open(chunk_all_comments_file, mode='w', newline='', encoding='utf-8') as all_csvfile, \
                 open(chunk_filtered_comments_file, mode='w', newline='', encoding='utf-8') as filtered_csvfile:
                
                all_csv_writer = csv.writer(all_csvfile)
                filtered_csv_writer = csv.writer(filtered_csvfile)
                
                all_csv_writer.writerow(['Submission Title', 'Submission Score', 'Submission URL', 'Submission Timestamp', 'Comment', 'Comment Score', 'Comment Timestamp'])
                filtered_csv_writer.writerow(['Submission Title', 'Submission Score', 'Submission URL', 'Submission Timestamp', 'Comment', 'Comment Score', 'Comment Timestamp'])

                all_chunk_posts = [] # collect all posts in this chunk
                while True:
                    url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&q={query}&after={after}&before={before}&sort=created_utc&sort_type=asc&size=1000"
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()['data']

                    if not data:
                        break  # No more data

                    all_chunk_posts.extend(data)

                    if len(data) < 1000:  # if less than 1000 posts, it means it is the last page.
                        break

                    before = int(data[-1]['created_utc'])  # update before parameter.

                for submission_data in all_chunk_posts:  # process all posts collected in this chunk.
                    submission_title = submission_data['title'].lower()
                    if any(keyword in submission_title for keyword in keywords):
                        total_posts += 1
                        submission_id = submission_data['id']
                        comments_url = f"https://api.pushshift.io/reddit/comment/search/?link_id={submission_id}&limit=1000"
                        comments_response = requests.get(comments_url)
                        comments_response.raise_for_status()
                        comments_data = comments_response.json()['data']

                        for comment_data in comments_data:
                            comment_body = comment_data['body'].lower()
                            comment_data_list = [
                                submission_data['title'],
                                submission_data['score'],
                                f"https://www.reddit.com{submission_data['permalink']}",
                                format_timestamp(submission_data['created_utc']),
                                comment_data['body'],
                                comment_data['score'],
                                format_timestamp(comment_data['created_utc'])
                            ]

                            all_csv_writer.writerow(comment_data_list)
                            total_comments += 1
                            total_comment_score += comment_data['score']

                            if any(keyword in comment_body for keyword in keywords):
                                filtered_csv_writer.writerow(comment_data_list)
                                total_filtered_comments += 1
                                total_filtered_comment_score += comment_data['score']

                        print(f"Collected {total_posts} posts from {current_start} to {current_end}")
                        time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Request error occurred for chunk {current_start} to {current_end}: {e}")
            time.sleep(10)
        except Exception as e:
            print(f"Error occurred for chunk {current_start} to {current_end}: {e}")
            break

        current_start = current_end

    # Calculate statistics and export to the statistics CSV
    with open(statistics_file, mode='w', newline='', encoding='utf-8') as stats_csvfile:
        stats_csv_writer = csv.writer(stats_csvfile)
        stats_csv_writer.writerow(['Statistic', 'Value'])
        
        stats_csv_writer.writerow(['Total Posts', total_posts])
        stats_csv_writer.writerow(['Total Comments', total_comments])
        stats_csv_writer.writerow(['Total Filtered Comments', total_filtered_comments])
        
        avg_comments_per_post = total_comments / total_posts if total_posts > 0 else 0
        stats_csv_writer.writerow(['Average Comments per Post', avg_comments_per_post])
        
        avg_comment_score = total_comment_score / total_comments if total_comments > 0 else 0
        avg_filtered_comment_score = total_filtered_comment_score / total_filtered_comments if total_filtered_comments > 0 else 0
        stats_csv_writer.writerow(['Average Comment Score', avg_comment_score])
        stats_csv_writer.writerow(['Average Filtered Comment Score', avg_filtered_comment_score])
        
        avg_filtered_comments_per_post = total_filtered_comments / total_posts if total_posts > 0 else 0
        stats_csv_writer.writerow(['Average Filtered Comments per Post', avg_filtered_comments_per_post])
        
        percentage_filtered = (total_filtered_comments / total_comments * 100) if total_comments > 0 else 0
        stats_csv_writer.writerow(['Percentage of Comments Filtered', percentage_filtered])

if __name__ == "__main__":
    search_posts_and_export_time_chunks(subreddit_name, keywords)

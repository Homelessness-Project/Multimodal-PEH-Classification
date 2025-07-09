import requests
import csv
import os
from datetime import datetime, timezone, timedelta
import time
import praw
CLIENT_ID = 'YOUR_CLIENT_ID'
CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
USER_AGENT = 'YOUR_USER_AGENT'

subreddit_name = "Scranton"

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

reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)

# Function to convert UTC timestamp to a readable format with timezone support
def format_timestamp(utc_timestamp):
    dt = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')

# Function to search posts, export data, and generate statistics
def search_posts_and_export(subreddit_name, keywords, start_date, end_date):
    total_posts = 0
    total_keyword_posts = 0  # To track posts with keywords
    total_comments = 0
    total_filtered_comments = 0
    total_comment_score = 0
    total_filtered_comment_score = 0

    processed_submissions = set()  # Track processed submission IDs
    processed_comments = set() # Track processed comment IDs

    with open(all_comments_file, mode='w', newline='', encoding='utf-8') as all_csvfile, \
         open(filtered_comments_file, mode='w', newline='', encoding='utf-8') as filtered_csvfile:

        all_csv_writer = csv.writer(all_csvfile)
        filtered_csv_writer = csv.writer(filtered_csvfile)

        all_csv_writer.writerow(['Submission Title', 'Submission Score', 'Submission URL', 'Submission Timestamp', 'Comment', 'Comment Score', 'Comment Timestamp'])
        filtered_csv_writer.writerow(['Submission Title', 'Submission Score', 'Submission URL', 'Submission Timestamp', 'Comment', 'Comment Score', 'Comment Timestamp'])

        # Convert start and end date to timestamp
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        #Using set for keywords
        keyword_set = set(keywords)
        
        for keyword in keyword_set:
            #(f"Searching for keyword: {keyword}")
            for submission in reddit.subreddit(subreddit_name).search(keyword, time_filter='all', limit=None):
                #print(f"Processing submission: {submission.title}")
                submission_timestamp = submission.created_utc
                # Only process submissions that fall within the given date range
                #print(f"Submission timestamp: {submission_timestamp}, Start timestamp: {start_timestamp}, End timestamp: {end_timestamp}")
                if start_timestamp <= submission_timestamp <= end_timestamp and submission.id not in processed_submissions:
                    processed_submissions.add(submission.id) #Add submission id to set so it wont be processed twice.
                    submission_title = submission.title.lower()
                    total_posts += 1

                    # Check if the submission contains any of the keywords
                    if any(kw in submission_title for kw in keyword_set):
                        total_keyword_posts += 1  # Count posts with keywords
                    submission_id = submission.id
                    submission_url = f"https://www.reddit.com{submission.permalink}"
                    submission_timestamp = format_timestamp(submission.created_utc)

                    submission.comments.replace_more(limit=0)  # Avoid 'MoreComments'
                    for comment in submission.comments.list():
                        if comment.id not in processed_comments:
                            processed_comments.add(comment.id)
                            comment_body = comment.body.lower()
                            comment_data_list = [
                                submission.title,
                                submission.score,
                                submission_url,
                                submission_timestamp,
                                comment.body,
                                comment.score,
                                format_timestamp(comment.created_utc)
                            ]

                            all_csv_writer.writerow(comment_data_list)
                            total_comments += 1
                            total_comment_score += comment.score

                            if any(kw in comment_body for kw in keyword_set):
                                filtered_csv_writer.writerow(comment_data_list)
                                total_filtered_comments += 1
                                total_filtered_comment_score += comment.score

                    print(f"Collected {total_posts} posts")
                    time.sleep(1)

    # Calculate statistics and export to the statistics CSV
    with open(statistics_file, mode='w', newline='', encoding='utf-8') as stats_csvfile:
        stats_csv_writer = csv.writer(stats_csvfile)
        stats_csv_writer.writerow(['Statistic', 'Value'])

        stats_csv_writer.writerow(['Start Date', start_date.strftime('%Y-%m-%d %H:%M:%S %Z')])
        stats_csv_writer.writerow(['End Date', end_date.strftime('%Y-%m-%d %H:%M:%S %Z')])

        stats_csv_writer.writerow(['Total Posts', total_posts])
        stats_csv_writer.writerow(['Total Keyword Posts', total_keyword_posts])
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

        # Calculate percentage of posts with keywords
        percentage_keyword_posts = (total_keyword_posts / total_posts * 100) if total_posts > 0 else 0
        stats_csv_writer.writerow(['Percentage of Posts with Keywords', percentage_keyword_posts])

if __name__ == "__main__":
    # Set the start and end date as required
    start_date = datetime(2015, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    search_posts_and_export(subreddit_name, keywords, start_date, end_date)

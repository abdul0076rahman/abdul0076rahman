import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Sample customer data
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'TotalSpend': [500, 1500, 200, 2500, 800, 1800, 300, 2200, 700, 1300],
    'Frequency': [5, 20, 2, 25, 10, 22, 3, 28, 7, 18],
    'LastPurchaseDaysAgo': [30, 5, 90, 3, 15, 7, 100, 2, 20, 10]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature selection and scaling
X = df[['TotalSpend', 'Frequency', 'LastPurchaseDaysAgo']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering to segment customers
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(X_scaled)

# Personalized recommendations based on segment
def recommend(segment):
    recommendations = {
        0: "Discount on premium products",
        1: "Exclusive membership benefits",
        2: "Personalized product suggestions based on browsing history"
    }
    return recommendations.get(segment, "General promotions")

df['Recommendation'] = df['Segment'].apply(recommend)

# Email marketing function (simulated)
def send_email(customer_id, recommendation):
    sender_email = "your_email@example.com"
    receiver_email = f"customer_{customer_id}@example.com"
    subject = "Exclusive Offer Just for You!"
    body = f"Hello Customer {customer_id},\n\nWe have a special offer for you: {recommendation}. Check it out now!"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Simulating an email server (replace with actual SMTP server details)
        server = smtplib.SMTP('smtp.example.com', 587)
        server.starttls()
        server.login(sender_email, "your_password")
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"Email sent to Customer {customer_id}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Simulate email notifications for each customer
for index, row in df.iterrows():
    send_email(row['CustomerID'], row['Recommendation'])

# Display results
print(df)

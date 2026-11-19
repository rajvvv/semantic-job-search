import pandas as pd

# Load the raw Kaggle CSV
df = pd.read_csv('data/DataScientist.csv')

# Keep first 1,000 rows
df = df.head(1000)

# Keep only the columns we need
df = df[['Job Title', 'Job Description', 'Company Name', 'Location']].copy()
df.columns = ['title', 'description', 'company', 'location']

# Clean all four columns
df['title']       = df['title'].astype(str).str.strip()
df['description'] = df['description'].astype(str).str.strip()
# Company Name contains "Hopper\n3.5" — take only the part before the newline
df['company']     = df['company'].astype(str).str.split('\n').str[0].str.strip()
df['location']    = df['location'].astype(str).str.strip()

# Drop rows with empty title or description (safety)
before = len(df)
df = df[(df['title'] != '') & (df['title'] != 'nan') &
        (df['description'] != '') & (df['description'] != 'nan')]
after = len(df)

# Write the clean file
df.to_csv('jobs_1000.csv', index=False)

print(f'Rows: {before} → {after} (dropped {before - after} empty)')
print(f'Saved to jobs_1000.csv')
print()
print('Sample (first 2 rows):')
for i in range(min(2, after)):
    row = df.iloc[i]
    print(f"  [{i}] title={row['title'][:50]!r}")
    print(f"      company={row['company'][:30]!r}")
    print(f"      location={row['location'][:30]!r}")
    print(f"      description={row['description'][:80]!r}...")

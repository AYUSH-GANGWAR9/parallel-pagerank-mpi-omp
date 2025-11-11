import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

csv = "data/scaling_results.csv"
if not os.path.exists(csv):
    print("No CSV found:", csv)
    raise SystemExit(1)

df = pd.read_csv(csv)
df['total_time_s'] = pd.to_numeric(df['total_time_s'], errors='coerce')
df_sorted = df.sort_values('procs').dropna(subset=['total_time_s']).reset_index(drop=True)
if df_sorted.empty:
    print("No numeric timing rows found in", csv)
    print(df)
    raise SystemExit(1)

t1 = df_sorted.loc[0,'total_time_s']
df_sorted['speedup'] = t1 / df_sorted['total_time_s']
df_sorted['efficiency'] = df_sorted['speedup'] / df_sorted['procs']

print(df_sorted)

plt.figure(figsize=(6,4))
plt.plot(df_sorted['procs'], df_sorted['total_time_s'], marker='o')
plt.xlabel('MPI processes')
plt.ylabel('Total time (s)')
plt.title('Strong scaling: total time')
plt.grid(True)
plt.savefig('data/total_time_plot.png', dpi=200)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(df_sorted['procs'], df_sorted['speedup'], marker='o', label='Measured')
plt.plot(df_sorted['procs'], df_sorted['procs'], '--', label='Ideal')
plt.xlabel('MPI processes')
plt.ylabel('Speedup')
plt.title('Strong scaling: speedup')
plt.legend()
plt.grid(True)
plt.savefig('data/speedup_plot.png', dpi=200)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(df_sorted['procs'], df_sorted['efficiency']*100, marker='o')
plt.xlabel('MPI processes')
plt.ylabel('Efficiency (%)')
plt.title('Strong scaling: efficiency')
plt.grid(True)
plt.savefig('data/efficiency_plot.png', dpi=200)
plt.close()

print("Saved plots to data/*.png")

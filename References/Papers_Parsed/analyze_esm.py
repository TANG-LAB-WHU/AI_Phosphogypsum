import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

xl = pd.ExcelFile('Environmental and health co-benefits for advanced phosphorus recovery_MOESM2_ESM.xlsx')

# Table 9 - Feedstock composition
print('='*100)
print('TABLE 9: Feedstock Composition')
print('='*100)
df9 = pd.read_excel(xl, sheet_name='Supplem.Table 9', header=0)
print(df9.to_string())

print('\n\n')

# Table 10 - P-fertilizer composition
print('='*100)
print('TABLE 10: Concentrated P-fertiliser Composition')
print('='*100)
df10 = pd.read_excel(xl, sheet_name='Supplem.Table 10', header=0)
print(df10.to_string())

print('\n\n')

# Table 20 - Unit costs
print('='*100)
print('TABLE 20: Unit Costs for Marketable Goods')
print('='*100)
df20 = pd.read_excel(xl, sheet_name='Supplem.Table 20', header=0)
print(df20.to_string())

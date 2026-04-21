import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os

print("loading basket matrix...")
basket = pd.read_csv(os.path.join("data", "basket_matrix.csv"), index_col=0)

# ensure boolean dtype
basket = basket.astype(bool)

print(f"basket shape: {basket.shape}")
print("running apriori... (this may take 1-2 minutes)")

frequent_itemsets = apriori(
    basket,
    min_support=0.02,       # item appears in at least 2% of transactions
    use_colnames=True,
    max_len=3               # max 3 items per rule
)

print(f"frequent itemsets found: {len(frequent_itemsets)}")

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

# filter for quality rules
rules = rules[rules["confidence"] >= 0.3]
rules = rules.sort_values("lift", ascending=False)

print(f"association rules generated: {len(rules)}")
print("\ntop 10 rules by lift:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).to_string())

# save
rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

rules.to_csv(os.path.join("data", "association_rules.csv"), index=False)
print("\nsaved rules to data/association_rules.csv")

import sys
import numpy as np
# Ensure we load from local src
sys.path.insert(0, ".")

from src.rl.rewards import reward_combo

print("Testing reward_combo return type...")
try:
    feats = np.random.rand(10, 512)
    sel = [0, 1, 2]
    ret = reward_combo(feats, sel, return_components=True)
    print(f"Return type: {type(ret)}")
    if isinstance(ret, tuple):
        print("✅ SUCCESS: Returned a tuple")
        print(f"Values: {ret}")
    else:
        print(f"❌ FAILURE: Returned {type(ret)} instead of tuple")
        print(f"Value: {ret}")

except Exception as e:
    print(f"❌ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()

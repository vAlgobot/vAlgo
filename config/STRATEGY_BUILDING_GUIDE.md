# 📊 vAlgo Strategy Building Guide

Complete step-by-step guide for building trading strategies using the hybrid config approach.

## 🎯 **Quick Overview**

The hybrid approach uses **3 main sheets** to build strategies:
1. **Entry_Conditions** - Define when to BUY/SELL
2. **Exit_Conditions** - Define when to exit positions  
3. **Strategy_Config** - Link entry/exit rules into complete strategies

## 📋 **Step-by-Step Strategy Building**

### **Step 1: Plan Your Strategy**

Before opening Excel, define:
- **Entry Signal**: When do you want to buy? (e.g., EMA crossover, RSI oversold)
- **Exit Signals**: When do you want to sell? (reverse signal, stop loss, take profit)
- **Risk Management**: How much loss can you take? What's your profit target?

**Example Plan:**
```
Strategy: EMA Crossover Scalping
Entry: Buy when EMA9 crosses above EMA21 with volume confirmation
Exit 1: Sell when price drops below EMA9 (signal reversal)  
Exit 2: Stop loss at 2% loss (risk management)
Exit 3: Take profit at 3% gain (profit target)
```

### **Step 2: Create Entry Rule (Entry_Conditions Sheet)**

Open Excel → Go to **Entry_Conditions** sheet → Add new row:

| Column | Value | Example |
|--------|--------|---------|
| Rule_Name | Unique name for your rule | `EMA_Crossover_Long` |
| Indicator_1 | First indicator to check | `EMA_9` |
| Operator_1 | Comparison operator | `crosses_above` |
| Value_1 | What to compare against | `EMA_21` |
| Logic | How to combine conditions | `AND` |
| Indicator_2 | Second indicator | `Volume` |
| Operator_2 | Second comparison | `>` |
| Value_2 | Second value | `AvgVolume_20` |
| Status | Turn rule on/off | `Active` |

**Result**: Buy when EMA_9 crosses above EMA_21 AND Volume > AvgVolume_20

### **Step 3: Create Exit Rules (Exit_Conditions Sheet)**

Create **3 separate exit rules** for the same strategy:

#### **Exit Rule 1: Signal Exit**
| Column | Value | Description |
|--------|--------|-------------|
| Rule_Name | `EMA_Signal_Exit` | Name for this exit |
| Exit_Type | `Signal` | Signal-based exit |
| Indicator | `Close` | Watch closing price |
| Operator | `<` | Less than |
| Value | `EMA_9` | EMA9 value |
| Status | `Active` | Turn on |

#### **Exit Rule 2: Stop Loss**
| Column | Value | Description |
|--------|--------|-------------|
| Rule_Name | `Stop_Loss_2_Percent` | Name for stop loss |
| Exit_Type | `Stop_Loss` | Risk management |
| Indicator | `PnL_Percent` | Track profit/loss % |
| Operator | `<=` | Less than or equal |
| Value | `-2` | 2% loss |
| Status | `Active` | Turn on |

#### **Exit Rule 3: Take Profit**
| Column | Value | Description |
|--------|--------|-------------|
| Rule_Name | `Take_Profit_3_Percent` | Name for profit target |
| Exit_Type | `Take_Profit` | Profit booking |
| Indicator | `PnL_Percent` | Track profit/loss % |
| Operator | `>=` | Greater than or equal |
| Value | `3` | 3% profit |
| Status | `Active` | Turn on |

### **Step 4: Link Everything (Strategy_Config Sheet)**

Create your complete strategy by linking entry and exit rules:

| Column | Value | Description |
|--------|--------|-------------|
| Strategy_Name | `My_EMA_Strategy` | Name your strategy |
| Entry_Rule | `EMA_Crossover_Long` | Reference your entry rule |
| Exit_Rule_1 | `EMA_Signal_Exit` | First exit condition |
| Exit_Rule_2 | `Stop_Loss_2_Percent` | Second exit condition |
| Exit_Rule_3 | `Take_Profit_3_Percent` | Third exit condition |
| Position_Size | `1000` | How much to buy |
| Risk_Per_Trade | `0.02` | 2% risk per trade |
| Max_Positions | `2` | Max simultaneous positions |
| Status | `Active` | Turn strategy on |

## 📚 **Complete Examples**

### **Example 1: VWAP Bounce Strategy**

**Goal**: Buy when price bounces off VWAP with volume

#### Entry_Conditions:
```
Rule_Name: VWAP_Bounce_Entry
Indicator_1: Close | Operator_1: > | Value_1: VWAP
Logic: AND
Indicator_2: Volume | Operator_2: > | Value_2: AvgVolume_20
Status: Active
```

#### Exit_Conditions:
```
Exit 1: VWAP_Exit
- Exit_Type: Signal | Indicator: Close | Operator: < | Value: VWAP

Exit 2: Quick_Stop  
- Exit_Type: Stop_Loss | Indicator: PnL_Percent | Operator: <= | Value: -1.5
```

#### Strategy_Config:
```
Strategy_Name: VWAP_Bounce_Scalping
Entry_Rule: VWAP_Bounce_Entry
Exit_Rule_1: VWAP_Exit
Exit_Rule_2: Quick_Stop
Position_Size: 500
Risk_Per_Trade: 0.015
```

### **Example 2: RSI Mean Reversion**

**Goal**: Buy oversold RSI, exit on rebound

#### Entry_Conditions:
```
Rule_Name: RSI_Oversold_Entry
Indicator_1: RSI_14 | Operator_1: < | Value_1: 30
Logic: AND  
Indicator_2: Close | Operator_2: > | Value_2: EMA_21
Status: Active
```

#### Exit_Conditions:
```
Exit 1: RSI_Rebound_Exit
- Exit_Type: Signal | Indicator: RSI_14 | Operator: > | Value: 70

Exit 2: Conservative_Stop
- Exit_Type: Stop_Loss | Indicator: PnL_Percent | Operator: <= | Value: -2
```

## 🔧 **Available Options Reference**

### **Indicators**
- **EMAs**: `EMA_9`, `EMA_21`, `EMA_50`
- **Price**: `Close`, `Open`, `High`, `Low`
- **Technical**: `RSI_14`, `VWAP`, `CPR_TC`, `CPR_BC`
- **Volume**: `Volume`, `AvgVolume_20`
- **System**: `PnL_Percent`, `Time`

### **Operators**
- **Basic**: `>`, `<`, `>=`, `<=`, `=`
- **Advanced**: `crosses_above`, `crosses_below`, `between`
- **Special**: `trail` (for trailing stops)

### **Exit Types**
- **Signal**: Price/indicator based exits
- **Stop_Loss**: Risk management exits
- **Take_Profit**: Profit booking exits  
- **Time_Based**: Time-based exits
- **Trailing_Stop**: Dynamic stop losses

### **Logic Operators**
- **AND**: Both conditions must be true
- **OR**: Either condition can be true

## 🎯 **Best Practices**

### **Entry Rules**
1. **Use volume confirmation** for breakout strategies
2. **Combine trend and momentum** indicators
3. **Avoid too many conditions** (max 2-3 per rule)

### **Exit Rules**
1. **Always have a stop loss** (risk management)
2. **Use multiple exit types** (signal + stop + target)
3. **Signal exits** should be opposite of entry logic

### **Position Sizing**
1. **Risk 1-2% per trade** maximum
2. **Start with smaller positions** while testing
3. **Limit concurrent positions** to manage risk

## 🚨 **Common Mistakes to Avoid**

1. **Complex Entry Rules**: Keep conditions simple and clear
2. **No Stop Loss**: Always define maximum acceptable loss  
3. **Unrealistic Targets**: Set achievable profit targets
4. **Overtrading**: Don't activate too many strategies at once
5. **No Volume Filter**: Include volume in breakout strategies

## 🧪 **Testing Your Strategy**

1. **Start with Status = Inactive** to test
2. **Validate** entry/exit rule names match exactly
3. **Check** position sizes are appropriate
4. **Paper trade first** before going live
5. **Monitor performance** and adjust as needed

---

**Remember**: Start simple, test thoroughly, and gradually add complexity as you gain confidence with the system!
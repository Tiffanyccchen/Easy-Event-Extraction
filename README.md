# Easy-Event-Extraction
A fast Chinese event extraction program 一個快速實現事件抽取的程式，繁體中文與簡體中文都支持

## 實現原理
  0. 若未分詞，用jieba分詞(速度快是優勢)
  1. 運用pylyp進行詞性標註跟依存句法分析
  2. 建立找中心語(Head)規則
  3. 建立找前後論元規則
  4. 建立事件抽取擴充版規則

## How to run

### Preinstall Packages

  1. opencc : a tool for converting traditional Chinese charaters into simplified Chinese charaters.
https://github.com/yichen0831/opencc-python

  2. jieba : an efficient Chinese tokenization tool.
https://github.com/fxsjy/jieba

  3. pyltp : a traditional Chinese language preprocessing utility containing tokenization, p.o.s. tagging, ne recognition , dependency parsing, etc. tools.
https://github.com/HIT-SCIR/pyltp

After Installation, you should have a ltp_data folder containing parser.model and pos.model ready for loading

### Run

```
>>> from Event_Extraction import Event_Extraction

1.沒有分詞
>>> events = Event_Extraction('三商美邦总经理杨棋材，请辞获准。', tra_sim = True, tokenize = False, expand = True)

2.有分詞(以空白間隔)
>>> events = Event_Extraction('三商 美邦 总经理 杨棋材 ，请辞 获准 。', tra_sim = True, tokenize = True, expand = True) 
--------
3.繁體中文輸入
>>> events = Event_Extraction('三商 美邦 總經理 楊棋材 ，请辭 獲准 。', tra_sim = False, tokenize = True, expand = False)
```

### Output

```
>>> events.events

1. standard version:
[['总经理','杨棋材'], ['请辞','获准']]

2. expand version:
[['美邦','总经理','杨棋材'], ['请辞','获准']]
```


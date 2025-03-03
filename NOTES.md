# NOTES

Notes related to the graduation project (i.e. evaluation) and implementation considerations.

## Queries

Datasets are collected in the `datesets` folder. It is important to consider which queries to evaluate on these datasets.

| Code | Dataset | Query | Notes |
|------|---------|-------|-------|
| T1 | `twitter` | `$.user.lang` | From GPJSON. Very simple query, only basic JSONPath support needed. |
| T2 | `twitter` | `$.user.lang[?(@ == 'nl')]` | From GPJSON. Modification of the basic query, but now with a filter. Filter is very selective, so few results. Could measure possible speedup with large skipping. |
| T3 | `twitter` | `$[*].entities.urls[*].url` | From JSONSKi and rsonpath. Uses the `*` for arrays. |
| B1 | `bestbuy` | `$.products[*].categoryPath[1:3].id` | From GPJSON and JSONSKi. Uses a slice (`1:3`) for arrays. |
| B2 | `bestbuy` | `$.products[*].videoChapters[*].chapter` | From JSONSKi and rsonpath. Does not add functionality over T3, but optional for extra data. |
| N1 | `nspl` | `$.meta.view.columns[*].name` | From JSONSKi and rsonpath. Does not add functionality over T3, but optional for extra data. |
| W1 | `walmart` | `$.items[*].bestMarketplacePrice.price` | From GPJSON. Does not add functionality over T3, but optional for extra data. |
| A1 | `ast` | `$..decl.name` | From rsonpath. Uses descendant (`..`) operator. Not sure if we will end up supporting this, but noting the query just in case. |
| A2 | `ast` | `$..inner..inner..type.qualType` | From rsonpath. Uses descendant (`..`) operator. Not sure if we will end up supporting this, but noting the query just in case. |

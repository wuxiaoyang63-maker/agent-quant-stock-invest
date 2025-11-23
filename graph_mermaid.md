---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__(<p>__start__</p>)
	extract_and_classify(extract_and_classify)
	filter_by_integrated_keywords(filter_by_integrated_keywords)
	select_supply_chain_industries(select_supply_chain_industries)
	fetch_supply_chain_stocks(fetch_supply_chain_stocks)
	fetch_industry_stocks(fetch_industry_stocks)
	analyze_industry(analyze_industry)
	generate_industry_report(generate_industry_report)
	analyze_stock(analyze_stock)
	save_report(save_report)
	__end__(<p>__end__</p>)
	__start__ --> extract_and_classify;
	analyze_industry --> generate_industry_report;
	analyze_stock --> save_report;
	extract_and_classify -. &nbsp;END&nbsp; .-> __end__;
	extract_and_classify -.-> filter_by_integrated_keywords;
	fetch_supply_chain_stocks --> analyze_industry;
	filter_by_integrated_keywords -. &nbsp;END&nbsp; .-> __end__;
	filter_by_integrated_keywords -.-> analyze_industry;
	filter_by_integrated_keywords -.-> analyze_stock;
	filter_by_integrated_keywords -.-> fetch_supply_chain_stocks;
	filter_by_integrated_keywords -.-> select_supply_chain_industries;
	generate_industry_report --> save_report;
	select_supply_chain_industries --> fetch_supply_chain_stocks;
	save_report --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

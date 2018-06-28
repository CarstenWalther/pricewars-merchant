# Data-Driven Merchant

This is a data-driven merchant for the [Price Wars platform](https://github.com/hpi-epic/pricewars).
The merchant make automated pricing and ordering decisions with the objective to maximize its expected long-term profit.
Demand learning is used to estimate customer demand based on historical market and sales data.
The estimated customer demand is used to calculate optimized joint ordering and pricing policies using dynamic programming.

The merchant was developed over the course of multiple problem scenarios.
The first scenario (scenario 0) is in a restricted environment.
Following scenarios become increasingly realistic but also require more sophisticated solutions.
The merchant implementation for each scenario can be found in the corresponding file (`merchant_scenarioX.py`).

|               | Ordering           | Demand Learning    | Pricing           | Competition        |
| ------------- |:------------------:|:------------------:|:-----------------:|:------------------:|
| Scenario 0    | :heavy_check_mark: | ❎                 | ❎                 | ❎                 |
| Scenario 1    | :heavy_check_mark: | ❎                 | ❎                 | ❎                 |
| Scenario 2    | :heavy_check_mark: | :heavy_check_mark: | ❎                 | ❎                 |
| Scenario 3    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | ❎                 |
| Scenario 4    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

The demand in scenario 0 is deterministic and known to the merchant.
In scenario 1, the demand is stochatic and the demand distribution is known.
In all other scenarios, the demand is stochastic but the merchant does not know the real demand distribution.

All scenarios consider trading with a single product type. However, the concept can easily be applied to multiple product types as long as there are no demand substitution effects between different products.

This repository is a fork of the official Price Wars [merchant repository](https://github.com/hpi-epic/pricewars) and thus contains example merchants.
The repository consists of three branches:
* **master**: Mirror of the upstream repository
* **dev**: Implementations of merchants for the problem scenarios
* **delayed_order**: Like the dev branch but the scenario 4 merchant considers order delay

## Setup

The merchant is written in Python and needs it to be installed.
After cloning the repository, install the necessary dependencies:
```
python3 -m pip install -r requirements.txt
```

## Run
Be sure that the Pricewars plattform is running.
After that you can run the merchant with:
```
python3 merchant_scenario4.py --port 5000
```
You might need to configure the URLs to the Price Wars services.
These can be set with `--marketplace`, `--producer` and `--kafka`.
Example: `python3 merchant_scenario4.py --port 5000 --marketplace http://192.168.47.1:8080 --producer http://192.168.47.1:3050 --kafka http://192.168.47.1:8001`
Run `python3 merchant_scenario4.py --help` to see all configuration parameters.

## Components

The data-driven merchant is structured as followed:
The initialization and the main loop is in `merchant_scenario4.py`.
Demand learning happens in `policy/demand_learning.py`. This is also where the demand learning features are defined.
Pricing and ordering policies are calculated in `policy/policy.py`.

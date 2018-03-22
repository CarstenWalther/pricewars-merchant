# Merchant

This repository contains the merchant-component of the Pricewars-simulation. The merchant represents a vendor trying to sell their products at the highest possible profit at the marketplace. As such, they also represent a certain pricing strategy that is to be tested against other pricing strategies. Such a pricing strategy can be either very simple and rule-driven, such as "always be 10 cents cheaper than your competitors" or data-driven and use machine learning strategies.

There is an example merchant available (in `merchant.py`) that implements the following strategies:

* Cheapest
  * always undercuts the competitors
* Two Bound
  * has a minimum and maximum price limit
  * keeps decreasing the price to maintain the 1st position in the price rank until the minimum limit is reached
  * then set the price back to the maximum price limit

There are three ways to build your own merchant:
* Use the example merchant and make changes to it (`merchant.py`)
* [Subclass the merchant base class and extend it](docs/Building%20a%20merchant%20using%20PricewarsMerchant.ipynb)
* Build the merchant from scratch and use the API implementation to communicate with the pricewars services (Take a look on how to use the [Marketplace and Producer API](docs/Handling%20products%20and%20offers.ipynb) and the [Kafka API](docs/Working%20with%20Kafka%20data.ipynb))

The meta repository containing general information can be found [here](https://github.com/hpi-epic/pricewars).

## Application Overview

| Repo |
|--- |
| [UI](https://github.com/hpi-epic/pricewars-mgmt-ui) |
| [Consumer](https://github.com/hpi-epic/pricewars-consumer) |
| [Producer](https://github.com/hpi-epic/pricewars-producer) |
| [Marketplace](https://github.com/hpi-epic/pricewars-marketplace) |
| [Merchant](https://github.com/hpi-epic/pricewars-merchant) |

## Requirements

The merchants are written in Python. Ensure to have [Python](https://www.python.org/) installed in version 3.5 or higher.

The code is type-hinted, so using an IDE (e.g. PyCharm) is helpful.

## Setup

After cloning the repository, install the necessary dependencies:
```
python3 -m pip install -r requirements.txt
```
Be sure that the Pricewars plattform is running.
After that you can run the example merchant with e.g. the cheapest strategy:
```
python3 merchant.py --port 5000 --strategy Cheapest
``` 
Run `python3 merchant.py --help` to see all configuration parameters.

## Concept

The merchant sells _products_ to _consumers_ on the _marketplace_. The interface to do so, is derived from real world online marketplaces. Thus, a merchant:

* has to **register** at the marketplace, which will grant him an identification and authorization token. This token can be mandatory for certain actions and is also used to keep track of all earnings and expenses of a merchant (e.g. in logs).
* can **buy products** form the producer. These are randomly chosen, so the merchant cannot actively choose products. This is due to the fact, that merchants should focus on pricing.
* can **put** products as **offers** on the marketplace, increase amount of products of an offer (**restock**) and reprice offers (requests are limited)
* needs to accept notifications about **sales** (as http request on `/sold`)

A merchant is both, a service and a client.
It needs to steadily interact on the marketplace and thus, should implement a kind of game loop or a series of scheduled events.
The merchant base class (`PricewarsMerchant`) offers a web server for sale/start/stop requests and an interaction loop.
Subclassing and overriding the merchant base class is all one needs to get a merchant started.

## Components

*Note*: Have a look at the samples in docs folder.
[This one](docs/Handling%20products%20and%20offers.ipynb) shows how to use the pricewars API to call the producer and marketplace.

This repository contains models and request APIs to ease the development of a merchant:

* Models
	* Offer
	* Order
	* Product
	* SoldOffer
* API to (using models according to [Swagger API](https://hpi-epic.github.io/pricewars/))
	* Marketplace (get/add/restock offers, register)
	* Producer (order products)
	* Kafka REST service (get market data as CSV)
* MerchantServer
	* provides web server to accept sold interface, updating settings and simple execution states (stop/stop)
* PricewarsMerchant
	* a base class that provides common functionality for merchants

### Using Market Data
To estimate demand, the merchant can access historical market data about different topics.
[This IPython Notebook](docs/Working%20with%20Kafka%20data.ipynb) and [this documentation](https://github.com/hpi-epic/pricewars-kafka-reverse-proxy#filtered-data-view-as-csv) explain how you can get and use that data.
Logging and data format is [documented here](https://github.com/hpi-epic/pricewars-marketplace#logging).
The data is returned in form of a pandas DataFrame.
# execution/alpaca_executor.py

import alpaca_trade_api as tradeapi
from config.secrets_config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

class AlpacaExecutor:
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

    def place_order(self, symbol: str, qty: float, side: str, order_type: str = 'market', time_in_force: str = 'gtc', 
                    limit_price: float = None, stop_price: float = None):
        """
        Place a buy or sell order.

        Parameters:
            symbol (str): Ticker symbol, e.g. 'AAPL'
            qty (float): Quantity to trade
            side (str): 'buy' or 'sell'
            order_type (str): 'market', 'limit', 'stop', or 'stop_limit'
            time_in_force (str): 'gtc', 'day', etc.
            limit_price (float, optional): Limit price for limit or stop_limit orders
            stop_price (float, optional): Stop price for stop or stop_limit orders
        """
        try:
            order_params = {
                'symbol': symbol,
                'qty': qty,
                'side': side.lower(),
                'type': order_type.lower(),
                'time_in_force': time_in_force.lower()
            }
            if order_type.lower() in ['limit', 'stop_limit'] and limit_price is not None:
                order_params['limit_price'] = limit_price
            if order_type.lower() in ['stop', 'stop_limit'] and stop_price is not None:
                order_params['stop_price'] = stop_price

            order = self.api.submit_order(**order_params)
            print(f"{side.capitalize()} order submitted: {order}")
            return order
        except Exception as e:
            print(f"Error placing {side} order: {e}")

    def cancel_all_orders(self):
        """
        Cancels all open/pending orders.
        """
        try:
            open_orders = self.api.list_orders(status='open')
            if not open_orders:
                print("No open orders to cancel.")
                return

            for order in open_orders:
                print(f"Cancelling order {order.id} for {order.symbol}...")
                self.api.cancel_order(order.id)
            
            print("All open orders cancelled.")
        except Exception as e:
            print(f"Error cancelling orders: {e}")

    def liquidate_all_positions(self):
        """
        Sells all current positions to fully liquidate the portfolio.
        """
        try:
            positions = self.api.list_positions()
            if not positions:
                print("No positions to liquidate.")
                return
            
            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                print(f"Submitting sell order for {qty} shares of {symbol}...")
                self.place_order(symbol=symbol, qty=qty, side='sell', order_type='market', time_in_force='gtc')
            
            print("All positions liquidation orders submitted.")
        except Exception as e:
            print(f"Error during liquidation: {e}")
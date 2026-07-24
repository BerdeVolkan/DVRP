from dvrpsim import Order


class LoggingOrder(Order):
    """
    Order-Subklasse, die t_request/t_pickup/t_delivery in
    model.run_log["orders"][order_id] schreibt (siehe dvrpsim.elements.order.Order
    Callbacks on_request/on_pickup/on_delivery, aufgerufen von request()/pickup()/deliver()).
    """

    def on_request(self) -> None:
        super().on_request()
        self.model.run_log["orders"].setdefault(self.id, {})["t_request"] = self.model.env.now

    def on_pickup(self) -> None:
        super().on_pickup()
        self.model.run_log["orders"].setdefault(self.id, {})["t_pickup"] = self.model.env.now

    def on_delivery(self) -> None:
        super().on_delivery()
        self.model.run_log["orders"].setdefault(self.id, {})["t_delivery"] = self.model.env.now

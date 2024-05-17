

def choose_broker(turnover, num_units):
    # Calculate charges for Zerodha
    zerodha_charges = min(0.005 * turnover, 100)

    # Calculate charges for Choice Broking
    choice_broking_charges = 0.002 * turnover

    # Calculate charges for Angel One
    angel_one_charges = min(0.005 * turnover, 0.05 * num_units)

    # Determine the minimum charges and the corresponding broker
    charges = {
        "Zerodha": zerodha_charges,
        "Choice Broking": choice_broking_charges
        "Angel One": angel_one_charges
    }
    
    print(charges)
    # Find the broker with the minimum charges
    best_broker = min(charges, key=charges.get)
    min_charges = charges[best_broker]

    return best_broker, min_charges

# Example usage:
turnover = float(input("Enter the turnover: "))
num_units = int(input("Enter the number of units: "))

best_broker, min_charges = choose_broker(turnover, num_units)
print(f"The best broker is {best_broker} with charges of {min_charges} rupees.")

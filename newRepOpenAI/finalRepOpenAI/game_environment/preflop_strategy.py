import csv

def load_preflop_chart(path='preflop_chart.csv'):
    chart = {}
    # Dummy load just to prove it works
    chart['AA'] = 'raise'
    chart['72'] = 'fold'
    return chart

def lookup_action(chart, hand):
    # Dummy lookup just to prove it works
    return chart.get(hand, 'call')

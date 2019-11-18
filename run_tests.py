from my_tests import anchor_mapping, speed_test


def run_acnhor_mapping():
    anchor_mapping.test_anchor_mapping()

def run_speed_test():
    speed_test.measure_mobilenet()
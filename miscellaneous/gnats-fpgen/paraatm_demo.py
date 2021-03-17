# GNATS sample
#
# Optimal Synthesis Inc.
#
# Oliver Chen
# 03.12.2020
#
# Demo of gate-to-gate trajectory simulation.
#
# The aircraft starts from the origin gate, goes through departing taxi plan, takes off, goes through different flight phases to the destination airport, and finally reaches the destination gate.


from gnats_gate_to_gate import GateToGate

sim=GateToGate()
data = sim(pushback_clearance_delay=10)['trajectory']

print('Number of trajectory points:',len(data))
print('Columns:\n',data.columns)


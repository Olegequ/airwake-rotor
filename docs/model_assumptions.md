# Model scope and assumptions · Olegequ

## Flow and decomposition

Coupling is one-way: the imposed airwake drives the rotor/body model, while
the rotor does not modify the stored CFD field. Mean flow is stored separately.
RVMD identifies temporal structure in a compressed state representation, then
recovers spatial modes from full-space regression. Compression and mode-count
choices require their own residual audits.

The study loader enforces `Lref = 13.716 m` and `Uref = 20 m/s` for
`St = f Lref / Uref`. Lower-level RVMD functions accept reference-scale
parameters. The delivered study file uses example timing and a placeholder hub
coordinate; neither establishes facts about a new dataset.

## Rotor, inflow and body

The blade-element model is quasi-steady and nonlinear. Airfoil-specific dynamic
stall is not included. Representative rotor/airframe values in the example
configuration include engineering estimates and are not manufacturer-certified
parameters. Supply measured values when quantitative interpretation requires
them.

The three-state Pitt–Peters disturbance formulation is restricted to its
declared near-hover regime. The body has nonlinear roll, pitch and heave with
yaw constrained. Hub aerodynamic moment and the moment transmitted to the body
are tracked separately. The primary force-line-only closure does not directly
transmit all aerodynamic disk bending moments.

The full workflow solves a fixed-body, azimuth-mean equilibrium in the CFD mean
flow, seeded by still-air momentum/BEM trim. This is not periodic-orbit
shooting. All deletion cases share that trim, initial inflow, controls and fixed
balancing reactions. There is no independent per-case mean subtraction or
retrim. The synthetic demo uses a shared still-air trim for its zero-mean
analytic field.

The study's ideal PD hover holder uses mass/inertia-scaled generalized forces.
It is an engineering assumption for unavailable aircraft AFCS information.
Actuator saturation and aircraft-specific control laws are not modelled. The
workflow includes gain-sensitivity checks and rejects declared domain-limit
violations.

## Interpreting a result

Full-space captured energy is computed from the residual of the reconstructed
mean-removed field. Nonorthogonal modal self-energies cannot be summed to
replace that gate. Response contribution R and unit-input-energy gain G have
different interpretations; report them separately. Leave-one-out deletion is
the primary estimator in the delivered configuration.

Associations among seven named aircraft configurations do not isolate rotor
diameter as a causal factor. A size-only claim needs controlled similarity
parameters and a virtual diameter sweep. Finite-window, closed-loop forced
response audits do not establish global stability or a flight-safety envelope.

The repository's tests establish selected numerical contracts and synthetic
examples. They do not validate the model against a real aircraft or redistribute
the original CFD evidence.

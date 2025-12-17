(********************
 * Color Sorting System - Structured Text example for Siemens S7-1200/1500
 *
 * This example demonstrates a simple state machine for sorting parts based on
 * color sensor input. It assumes discrete gates for RED, GREEN, and BLUE bins,
 * a conveyor motor, and a reject cylinder for unknown colors. Adapt the I/O
 * mapping in your hardware configuration and OB1 as needed.
 ********************)

TYPE eColor : (COLOR_NONE, COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_UNKNOWN);
END_TYPE

TYPE eSorterState : (STATE_IDLE, STATE_FEEDING, STATE_ROUTING, STATE_REJECT, STATE_FAULT);
END_TYPE

FUNCTION_BLOCK FB_ColorSorter
VAR_INPUT
    Start         : BOOL;      // Start command
    Stop          : BOOL;      // Stop command
    Reset         : BOOL;      // Fault reset
    PartPresent   : BOOL;      // Part detected at sensor station
    SensorHealthy : BOOL;      // Color sensor diagnostic bit
    DetectedColor : eColor;    // Enum from color sensor block
    RedBinReady   : BOOL;      // Bin full/ready interlocks
    GreenBinReady : BOOL;
    BlueBinReady  : BOOL;
END_VAR

VAR_OUTPUT
    ConveyorMotor : BOOL;
    GateRed       : BOOL;
    GateGreen     : BOOL;
    GateBlue      : BOOL;
    RejectCylinder: BOOL;
    Fault         : BOOL;
    CurrentState  : eSorterState;
END_VAR

VAR
    tFeed   : TON;
    tRoute  : TON;
    tReject : TON;
    nextState : eSorterState := STATE_IDLE;
END_VAR

// Main execution
BEGIN
    // Default outputs
    ConveyorMotor := FALSE;
    GateRed := FALSE;
    GateGreen := FALSE;
    GateBlue := FALSE;
    RejectCylinder := FALSE;
    Fault := FALSE;

    // Fault detection
    IF Stop THEN
        nextState := STATE_IDLE;
    END_IF;

    IF NOT SensorHealthy THEN
        Fault := TRUE;
        nextState := STATE_FAULT;
    END_IF;

    CASE nextState OF
        STATE_IDLE:
            CurrentState := STATE_IDLE;
            IF Start AND NOT Stop THEN
                ConveyorMotor := TRUE;
                tFeed(IN := PartPresent, PT := T#1s);
                IF tFeed.Q THEN
                    nextState := STATE_ROUTING;
                END_IF;
            END_IF;

        STATE_ROUTING:
            CurrentState := STATE_ROUTING;
            ConveyorMotor := TRUE;
            tRoute(IN := PartPresent, PT := T#500ms);

            IF NOT PartPresent THEN
                // No part, return to idle
                nextState := STATE_IDLE;
            ELSIF DetectedColor = COLOR_RED AND RedBinReady THEN
                GateRed := TRUE;
                tRoute(IN := TRUE, PT := T#1s);
                IF tRoute.Q THEN
                    nextState := STATE_IDLE;
                END_IF;
            ELSIF DetectedColor = COLOR_GREEN AND GreenBinReady THEN
                GateGreen := TRUE;
                tRoute(IN := TRUE, PT := T#1s);
                IF tRoute.Q THEN
                    nextState := STATE_IDLE;
                END_IF;
            ELSIF DetectedColor = COLOR_BLUE AND BlueBinReady THEN
                GateBlue := TRUE;
                tRoute(IN := TRUE, PT := T#1s);
                IF tRoute.Q THEN
                    nextState := STATE_IDLE;
                END_IF;
            ELSE
                // Unknown color or bin not ready
                nextState := STATE_REJECT;
            END_IF;

        STATE_REJECT:
            CurrentState := STATE_REJECT;
            RejectCylinder := TRUE;
            tReject(IN := TRUE, PT := T#750ms);
            IF tReject.Q THEN
                nextState := STATE_IDLE;
            END_IF;

        STATE_FAULT:
            CurrentState := STATE_FAULT;
            Fault := TRUE;
            ConveyorMotor := FALSE;
            GateRed := FALSE;
            GateGreen := FALSE;
            GateBlue := FALSE;
            RejectCylinder := FALSE;

            IF Reset THEN
                Fault := FALSE;
                nextState := STATE_IDLE;
            END_IF;

        ELSE
            // Fallback
            nextState := STATE_IDLE;
    END_CASE;
END_FUNCTION_BLOCK

(*****************************
 * Example call in OB1 (pseudo)
 *****************************
 // Declare an instance of FB_ColorSorter
 VAR
     sorter : FB_ColorSorter;
 END_VAR

 sorter(
     Start := M0.0,
     Stop := M0.1,
     Reset := M0.2,
     PartPresent := I0.0,
     SensorHealthy := I0.1,
     DetectedColor := ColorSensorOutput,
     RedBinReady := I0.2,
     GreenBinReady := I0.3,
     BlueBinReady := I0.4,
     ConveyorMotor => Q0.0,
     GateRed => Q0.1,
     GateGreen => Q0.2,
     GateBlue => Q0.3,
     RejectCylinder => Q0.4,
     Fault => Q0.5,
     CurrentState => MD100
 );
*)

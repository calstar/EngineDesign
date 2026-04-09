export type ComponentType =
  | 'RTD' | 'PT' | 'PG' | 'LC' | 'TC'   // sensors
  | 'MAN' | 'ROT' | 'SOL'               // valves
  | 'PR' | 'RV' | 'CV' | 'QD'          // flow control
  | 'TANK' | 'INJECTOR';                 // hardware

export type FluidType = 'fuel' | 'lox' | 'pressurant' | 'default';

export const FLUID_COLORS: Record<FluidType, string> = {
  fuel:       '#f97316', // orange
  lox:        '#60a5fa', // blue
  pressurant: '#ef4444', // red
  default:    '#94a3b8', // gray
};

export interface PIDNodeData {
  componentType: ComponentType;
  label: string;
  fluidType?: FluidType;
  notes?: string;
}

export interface ComponentDef {
  type: ComponentType;
  label: string;
  fullName: string;
  group: 'Sensors' | 'Valves' | 'Flow Control' | 'Hardware';
}

export const COMPONENT_DEFS: ComponentDef[] = [
  // Sensors
  { type: 'RTD',      label: 'RTD_#',   fullName: 'Resistance Temperature Detector', group: 'Sensors' },
  { type: 'PT',       label: 'PT_#',    fullName: 'LOX/Eth Pressure Transducer',     group: 'Sensors' },
  { type: 'PG',       label: 'PG_#',    fullName: 'Pressure Gauge',                  group: 'Sensors' },
  { type: 'LC',       label: 'LC_#',    fullName: 'Load Cell',                       group: 'Sensors' },
  { type: 'TC',       label: 'TC_#',    fullName: 'Thermocouple',                    group: 'Sensors' },
  // Valves
  { type: 'MAN',      label: 'MAN_#',   fullName: 'Ball Valve (Manual)',              group: 'Valves' },
  { type: 'ROT',      label: 'ROT_#',   fullName: 'Ball Valve (Rotary)',              group: 'Valves' },
  { type: 'SOL',      label: 'SOL_#',   fullName: 'Solenoid Valve',                  group: 'Valves' },
  // Flow Control
  { type: 'PR',       label: 'PR_#',    fullName: 'Pressure Regulator',              group: 'Flow Control' },
  { type: 'RV',       label: 'RV_#',    fullName: 'Relief Valve',                    group: 'Flow Control' },
  { type: 'CV',       label: 'CV_#',    fullName: 'Kero/LOX Check Valve',            group: 'Flow Control' },
  { type: 'QD',       label: '[F]QD',   fullName: 'Quick Disconnect (Face Seal)',     group: 'Flow Control' },
  // Hardware
  { type: 'TANK',     label: 'TANK',    fullName: 'Tank / COPV',                     group: 'Hardware' },
  { type: 'INJECTOR', label: 'INJ',     fullName: 'Injector',                        group: 'Hardware' },
];

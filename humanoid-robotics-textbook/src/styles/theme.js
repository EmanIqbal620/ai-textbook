// Base theme with soft pastel colors
export const lightTheme = {
  // Pastel colors
  primary: '#A5D8FF',      // Soft blue
  secondary: '#B5EAD7',    // Soft mint
  accent: '#FFDAC1',       // Soft peach
  background: '#F8F9FA',   // Very light gray
  surface: '#FFFFFF',      // White
  text: '#212529',         // Dark gray
  textSecondary: '#6C757D', // Medium gray
  success: '#B5EAD7',      // Soft mint (for positive actions)
  warning: '#FFDAC1',      // Soft peach (for warnings)
  error: '#FF9AA2',        // Soft pink (for errors)

  // Specific platform colors
  ros2: '#6ab0f3',         // Blue for ROS 2
  gazebo: '#77cc6d',       // Green for Gazebo
  unity: '#c89850',        // Orange for Unity
  nvidia: '#76b900',       // Green for NVIDIA Isaac
  vla: '#a067ab',          // Purple for VLA
  capstone: '#f9c56b',     // Yellow for Capstone

  // Typography
  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  fontSize: {
    small: '0.875rem',
    medium: '1rem',
    large: '1.25rem',
    xlarge: '1.5rem',
    xxlarge: '2rem'
  },

  // Spacing
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    xxl: '3rem'
  },

  // Breakpoints
  breakpoints: {
    sm: '576px',
    md: '768px',
    lg: '992px',
    xl: '1200px'
  },

  // Shadows
  shadow: {
    light: '0 1px 3px rgba(0,0,0,0.1)',
    medium: '0 4px 6px rgba(0,0,0,0.1)',
    heavy: '0 10px 15px rgba(0,0,0,0.1)'
  },

  // Borders
  border: {
    radius: '8px',
    width: '1px'
  }
};

export const darkTheme = {
  // Dark mode pastel colors (darker versions)
  primary: '#265D8F',      // Darker soft blue
  secondary: '#3A7D6B',    // Darker soft mint
  accent: '#8F6D4B',       // Darker soft peach
  background: '#121212',   // Dark background
  surface: '#1E1E1E',      // Dark surface
  text: '#E1E1E1',         // Light gray
  textSecondary: '#A0A0A0', // Medium light gray
  success: '#3A7D6B',      // Darker soft mint
  warning: '#8F6D4B',      // Darker soft peach
  error: '#8F4C52',        // Darker soft pink

  // Specific platform colors (darker)
  ros2: '#3a6ca3',
  gazebo: '#4a7a4a',
  unity: '#8a6a3a',
  nvidia: '#4a7a00',
  vla: '#7a4a7a',
  capstone: '#a98a3b',

  // Typography (same as light theme)
  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  fontSize: {
    small: '0.875rem',
    medium: '1rem',
    large: '1.25rem',
    xlarge: '1.5rem',
    xxlarge: '2rem'
  },

  // Spacing (same as light theme)
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    xxl: '3rem'
  },

  // Breakpoints (same as light theme)
  breakpoints: {
    sm: '576px',
    md: '768px',
    lg: '992px',
    xl: '1200px'
  },

  // Shadows (darker)
  shadow: {
    light: '0 1px 3px rgba(0,0,0,0.3)',
    medium: '0 4px 6px rgba(0,0,0,0.3)',
    heavy: '0 10px 15px rgba(0,0,0,0.3)'
  },

  // Borders (same as light theme)
  border: {
    radius: '8px',
    width: '1px'
  }
};
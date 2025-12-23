import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar structure for the Physical AI & Humanoid Robotics textbook
  tutorialSidebar: [
    'book-intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/index',
        'module-1-ros2/week-1',
        'module-1-ros2/week-2',
        'module-1-ros2/week-3',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-simulation/index',
        'module-2-simulation/week-4',
        'module-2-simulation/week-5',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-ai-brain/index',
        'module-3-ai-brain/week-6',
        'module-3-ai-brain/week-7',
        'module-3-ai-brain/week-8',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/index',
        'module-4-vla/week-9',
        'module-4-vla/week-10',
        'module-4-vla/week-11',
        'module-4-vla/week-12',
        'module-4-vla/week-13',
      ],
    },
    {
      type: 'category',
      label: 'Module 5: Hardware Requirements',
      items: [
        'module-5-hardware/index',
        'module-5-hardware/hardware-specifications',
      ],
    },
    {
      type: 'category',
      label: 'Module 6: Assessment Guidelines',
      items: [
        'module-6-assessment/index',
        'module-6-assessment/assessment-methods',
      ],
    },
    {
      type: 'category',
      label: 'Resources',
      items: [
        'prerequisites',
        'glossary',
        'bibliography',
        'additional-resources',
      ],
    },
  ],
};

export default sidebars;

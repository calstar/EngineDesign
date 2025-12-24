import { useState, useEffect } from 'react';
import { DesignRequirements } from './DesignRequirements';
import { Layer1Optimization } from './Layer1Optimization';
import { 
  saveDesignRequirements, 
  getDesignRequirements
} from '../api/client';
import type { 
  DesignRequirements as DesignRequirementsType,
  EngineConfig 
} from '../api/client';

interface OptimizerProps {
  config: EngineConfig | null;
}

type SubTab = 'requirements' | 'layer1';

export function Optimizer({ config }: OptimizerProps) {
  const [activeSubTab, setActiveSubTab] = useState<SubTab>('requirements');
  const [requirements, setRequirements] = useState<DesignRequirementsType | null>(null);
  const [saveStatus, setSaveStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  // Load requirements on mount
  useEffect(() => {
    loadRequirements();
  }, []);

  const loadRequirements = async () => {
    const response = await getDesignRequirements();
    if (response.data?.requirements) {
      setRequirements(response.data.requirements);
    }
  };

  const handleSave = async (reqs: DesignRequirementsType) => {
    const response = await saveDesignRequirements(reqs);
    
    if (response.error) {
      setSaveStatus({ type: 'error', message: response.error });
    } else if (response.data) {
      setRequirements(response.data.requirements);
      setSaveStatus({ type: 'success', message: 'Design requirements saved successfully!' });
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSaveStatus(null);
      }, 3000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Main Header */}
      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl p-6">
        <h1 className="text-3xl font-bold text-[var(--color-text-primary)] mb-2">🚀 Engine Design Optimization</h1>
        <p className="text-[var(--color-text-secondary)]">
          <strong>Goal:</strong> Size optimal injector and chamber geometry to meet your:
        </p>
        <ul className="text-[var(--color-text-secondary)] list-disc list-inside mt-2 space-y-1">
          <li><strong>Stability margins</strong> (chugging, acoustic, feed system)</li>
          <li><strong>Flight performance</strong> (altitude, payload capacity)</li>
          <li><strong>System constraints</strong> (weight, size, manufacturing)</li>
        </ul>
      </div>

      {/* Save Status */}
      {saveStatus && (
        <div className={`rounded-xl p-4 border ${
          saveStatus.type === 'success' 
            ? 'bg-green-500/10 border-green-500/30 text-green-400'
            : 'bg-red-500/10 border-red-500/30 text-red-400'
        }`}>
          <p className="font-semibold">
            {saveStatus.type === 'success' ? '✅' : '❌'} {saveStatus.message}
          </p>
        </div>
      )}

      {/* Sub-tabs Navigation */}
      <nav className="flex gap-2 border-b border-[var(--color-border)]">
        <button
          onClick={() => setActiveSubTab('requirements')}
          className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
            activeSubTab === 'requirements'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-border)]'
          }`}
        >
          📋 Design Requirements
        </button>
        <button
          onClick={() => setActiveSubTab('layer1')}
          className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
            activeSubTab === 'layer1'
              ? 'border-purple-500 text-purple-400'
              : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-border)]'
          }`}
        >
          🔧 Layer 1: Static Optimization
        </button>
      </nav>

      {/* Sub-tab Content */}
      <div className="mt-6">
        {activeSubTab === 'requirements' && (
          <DesignRequirements config={config} onSave={handleSave} />
        )}
        {activeSubTab === 'layer1' && (
          <Layer1Optimization requirements={requirements} />
        )}
      </div>
    </div>
  );
}


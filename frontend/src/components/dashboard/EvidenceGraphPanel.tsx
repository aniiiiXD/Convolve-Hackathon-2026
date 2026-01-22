"use client";

import { cn } from "@/lib/utils";
import { motion } from "motion/react";
import { EvidenceGraph, GraphNode, NodeType } from "@/types";
import { Card, CardHeader } from "@/components/ui";
import {
  FileText,
  Search,
  Brain,
  CheckCircle,
  ArrowRight,
  GitBranch,
} from "lucide-react";

interface EvidenceGraphPanelProps {
  graph: EvidenceGraph;
  className?: string;
  onNodeClick?: (node: GraphNode) => void;
}

const nodeIcons: Record<NodeType, React.ElementType> = {
  evidence: FileText,
  finding: Search,
  reasoning: Brain,
  conclusion: CheckCircle,
  recommendation: ArrowRight,
};

const nodeColors: Record<NodeType, { bg: string; border: string; text: string }> = {
  evidence: {
    bg: "bg-scan-ultrasound/10",
    border: "border-scan-ultrasound/50",
    text: "text-scan-ultrasound",
  },
  finding: {
    bg: "bg-vital-info/10",
    border: "border-vital-info/50",
    text: "text-vital-info",
  },
  reasoning: {
    bg: "bg-scan-mri/10",
    border: "border-scan-mri/50",
    text: "text-scan-mri",
  },
  conclusion: {
    bg: "bg-vital-pulse/10",
    border: "border-vital-pulse/50",
    text: "text-vital-pulse",
  },
  recommendation: {
    bg: "bg-vital-warning/10",
    border: "border-vital-warning/50",
    text: "text-vital-warning",
  },
};

export default function EvidenceGraphPanel({
  graph,
  className,
  onNodeClick,
}: EvidenceGraphPanelProps) {
  // Group nodes by type for visualization
  const nodesByType = graph.nodes.reduce((acc, node) => {
    if (!acc[node.type]) acc[node.type] = [];
    acc[node.type].push(node);
    return acc;
  }, {} as Record<NodeType, GraphNode[]>);

  const typeOrder: NodeType[] = [
    "evidence",
    "finding",
    "reasoning",
    "conclusion",
    "recommendation",
  ];

  return (
    <Card className={cn("p-6", className)}>
      <CardHeader
        title="Evidence Graph"
        subtitle="Reasoning chain from evidence to conclusion"
        action={
          <div className="flex items-center gap-2 text-vital-pulse">
            <GitBranch className="h-4 w-4" />
            <span className="text-xs font-mono">
              {graph.nodes.length} nodes, {graph.edges.length} connections
            </span>
          </div>
        }
      />

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mb-6 pb-4 border-b border-clinical-border/30">
        {typeOrder.map((type) => {
          const Icon = nodeIcons[type];
          const colors = nodeColors[type];
          return (
            <div key={type} className="flex items-center gap-1.5">
              <div className={cn("p-1 rounded", colors.bg)}>
                <Icon className={cn("h-3 w-3", colors.text)} />
              </div>
              <span className="text-xs text-clinical-muted capitalize">
                {type}
              </span>
            </div>
          );
        })}
      </div>

      {/* Graph Visualization - Layered approach */}
      <div className="space-y-6">
        {typeOrder.map((type, typeIndex) => {
          const nodes = nodesByType[type];
          if (!nodes || nodes.length === 0) return null;

          return (
            <motion.div
              key={type}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: typeIndex * 0.1 }}
            >
              <div className="flex items-center gap-2 mb-3">
                <div
                  className={cn(
                    "w-8 h-px",
                    nodeColors[type].bg.replace("/10", "")
                  )}
                />
                <span className="text-xs text-clinical-muted uppercase tracking-wider">
                  {type}
                </span>
                <div className="flex-1 h-px bg-clinical-border/30" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 pl-4">
                {nodes.map((node, nodeIndex) => (
                  <EvidenceNode
                    key={node.id}
                    node={node}
                    index={nodeIndex}
                    onClick={() => onNodeClick?.(node)}
                  />
                ))}
              </div>

              {/* Connection lines to next layer */}
              {typeIndex < typeOrder.length - 1 && nodesByType[typeOrder[typeIndex + 1]] && (
                <div className="flex justify-center py-2">
                  <motion.div
                    initial={{ scaleY: 0 }}
                    animate={{ scaleY: 1 }}
                    transition={{ delay: typeIndex * 0.1 + 0.2 }}
                    className="w-px h-8 bg-gradient-to-b from-clinical-border to-transparent"
                  />
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Summary */}
      {graph.nodes.length > 0 && (
        <div className="mt-6 pt-4 border-t border-clinical-border/30">
          <div className="flex items-center justify-between text-sm">
            <span className="text-clinical-muted">Reasoning depth:</span>
            <span className="font-mono text-vital-pulse">
              {typeOrder.filter((t) => nodesByType[t]?.length > 0).length} layers
            </span>
          </div>
        </div>
      )}
    </Card>
  );
}

interface EvidenceNodeProps {
  node: GraphNode;
  index: number;
  onClick?: () => void;
}

function EvidenceNode({ node, index, onClick }: EvidenceNodeProps) {
  const Icon = nodeIcons[node.type];
  const colors = nodeColors[node.type];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ scale: 1.02 }}
      onClick={onClick}
      className={cn(
        "p-3 rounded-xl border cursor-pointer transition-all",
        colors.bg,
        colors.border,
        "hover:shadow-lg"
      )}
    >
      <div className="flex items-start gap-2">
        <Icon className={cn("h-4 w-4 mt-0.5 flex-shrink-0", colors.text)} />
        <div className="min-w-0">
          <p className="text-sm font-medium text-clinical-white line-clamp-1">
            {node.label}
          </p>
          <p className="text-xs text-clinical-muted line-clamp-2 mt-0.5">
            {node.content}
          </p>
          {node.confidence !== undefined && (
            <div className="mt-2 flex items-center gap-1.5">
              <div className="flex-1 h-1 rounded-full bg-midnight-50 overflow-hidden">
                <div
                  className={cn("h-full rounded-full", colors.text.replace("text-", "bg-"))}
                  style={{ width: `${node.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs font-mono text-clinical-muted">
                {Math.round(node.confidence * 100)}%
              </span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

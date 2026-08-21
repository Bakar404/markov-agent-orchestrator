"use client";

import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useState } from "react";

import { PixelSprite } from "@/components/pixel/PixelSprite";
import { api } from "@/lib/api";
import { useGame } from "@/lib/store";
import type { InteractionGraph } from "@/lib/types";

type AgentNodeData = {
  label: string;
  agentId: string;
  color: string;
  active: boolean;
  invocations: number;
  reward: number;
};

function AgentNode({ data }: NodeProps) {
  const node = data as AgentNodeData;
  return (
    <div
      className={`border-2 bg-ink px-2 py-2 text-center ${node.active ? "shadow-glow" : ""}`}
      style={{ borderColor: node.active ? node.color : "#3b356b", width: 118 }}
    >
      <Handle type="target" position={Position.Top} />
      <div className="flex justify-center">
        <PixelSprite id={node.agentId} size={44} dim={!node.active && node.invocations === 0} />
      </div>
      <p className="mt-1 font-pixel text-3xs" style={{ color: node.color }}>
        {node.label}
      </p>
      <p className="font-mono text-3xs text-edge">
        x{node.invocations} ·{" "}
        <span className={node.reward >= 0 ? "text-lime" : "text-crimson"}>
          {node.reward >= 0 ? "+" : ""}
          {node.reward.toFixed(2)}
        </span>
      </p>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = { agent: AgentNode };

export function GraphView() {
  const { run, metrics, activeAgents, lastStep } = useGame();
  const [graph, setGraph] = useState<InteractionGraph | null>(null);

  useEffect(() => {
    if (!run) return;
    let cancelled = false;
    api
      .interactionGraph(run.id)
      .then((data) => {
        if (!cancelled) setGraph(data);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [run, lastStep]);

  const nodes = useMemo<Node[]>(() => {
    if (!run) return [];
    const rewardOf = new Map((metrics?.per_agent ?? []).map((a) => [a.agent_id, a.reward]));

    const agentNodes: Node[] = run.agents.map((agent) => ({
      id: agent.id,
      type: "agent",
      position: agent.canvas_position,
      data: {
        label: agent.label.replace(" Agent", "").toUpperCase(),
        agentId: agent.id,
        color: agent.color,
        active: activeAgents.includes(agent.id),
        invocations: run.state.agent_history?.[agent.id]?.invocations ?? 0,
        reward: rewardOf.get(agent.id) ?? 0,
      } satisfies AgentNodeData,
    }));

    agentNodes.push({
      id: "orchestrator",
      type: "agent",
      position: { x: 435, y: 40 },
      data: {
        label: "CORE",
        agentId: "orchestrator",
        color: "#7bf7c4",
        active: activeAgents.length > 0,
        invocations: run.step_count,
        reward: run.cumulative_reward,
      } satisfies AgentNodeData,
    });

    return agentNodes;
  }, [run, metrics, activeAgents]);

  const edges = useMemo<Edge[]>(() => {
    if (!graph) return [];
    const peak = Math.max(...graph.edges.map((e) => e.count), 1);

    return graph.edges.map((edge) => {
      const live =
        activeAgents.includes(edge.source) || activeAgents.includes(edge.target);
      const isReport = edge.kinds.includes("report");
      return {
        id: `${edge.source}->${edge.target}`,
        source: edge.source,
        target: edge.target,
        animated: live,
        label: `${edge.count}× p̄=${edge.mean_weight.toFixed(2)}`,
        labelStyle: { fill: "#8f89c9", fontSize: 8, fontFamily: "IBM Plex Mono, monospace" },
        labelBgStyle: { fill: "#0b0a14" },
        style: {
          stroke: live ? "#7bf7c4" : isReport ? "#a78bfa" : "#3b356b",
          strokeWidth: 1 + (edge.count / peak) * 3,
        },
        markerEnd: { type: MarkerType.ArrowClosed, color: live ? "#7bf7c4" : "#3b356b" },
      } satisfies Edge;
    });
  }, [graph, activeAgents]);

  if (!run) return null;

  return (
    <section className="panel" style={{ height: "34rem" }}>
      <div className="flex items-baseline justify-between px-4 py-2">
        <h3 className="font-pixel text-2xs text-phosphor">AGENT INTERACTION GRAPH</h3>
        <span className="font-mono text-3xs text-edge">
          {edges.length} channels · edge width = message volume, label = mean probability weight
        </span>
      </div>
      <div style={{ height: "calc(100% - 2.5rem)" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          proOptions={{ hideAttribution: true }}
          nodesDraggable
          minZoom={0.3}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#221f42" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </section>
  );
}

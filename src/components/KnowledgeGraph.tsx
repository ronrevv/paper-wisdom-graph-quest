
import { useEffect, useRef, useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { NetworkIcon, Loader2 } from "lucide-react";

type Node = {
  id: number;
  label: string;
  group?: number;
};

type Edge = {
  from: number;
  to: number;
  label?: string;
};

type GraphData = {
  nodes: Node[];
  edges: Edge[];
};

interface KnowledgeGraphProps {
  data: GraphData;
}

export const KnowledgeGraph = ({ data }: KnowledgeGraphProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("Initializing graph visualization...");

  // This function would be replaced with actual visualization code using a library like vis.js or cytoscape
  // For this demo, we'll simulate it with a timeout
  useEffect(() => {
    const timer = setTimeout(() => {
      if (data.nodes.length > 0) {
        setLoading(false);
      }
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [data]);

  // In a real implementation, this would render the actual graph using a library
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium flex items-center gap-2">
          <NetworkIcon className="h-5 w-5" />
          Knowledge Graph Visualization
        </h3>
      </div>
      
      <Card>
        <CardContent className="p-0">
          <div 
            ref={containerRef}
            className="h-[500px] rounded-md bg-secondary/30 flex flex-col items-center justify-center"
          >
            {loading ? (
              <div className="text-center space-y-2">
                <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto" />
                <p className="text-sm text-muted-foreground">{message}</p>
              </div>
            ) : (
              <div className="p-4 w-full h-full">
                <div className="mb-4">
                  <p className="text-sm text-muted-foreground">
                    Displaying a graph with {data.nodes.length} nodes and {data.edges.length} edges.
                  </p>
                </div>
                
                {/* This would be replaced with actual graph rendering */}
                <div className="bg-secondary/50 p-4 rounded-md h-[400px] flex items-center justify-center">
                  <div className="text-center">
                    <NetworkIcon className="h-16 w-16 mx-auto text-primary/60 mb-4" />
                    <p className="text-sm max-w-md">
                      In a complete implementation, this would display an interactive knowledge graph 
                      visualization showing entities and relationships extracted from the research papers.
                    </p>
                    <div className="mt-4">
                      <p className="text-xs text-muted-foreground">
                        Nodes: {data.nodes.map(node => node.label).join(', ')}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

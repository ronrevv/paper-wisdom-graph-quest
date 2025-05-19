
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Layers, Database, Network, Cpu, LineChart, 
  FileText, Loader2, RefreshCw
} from "lucide-react";

export const InfoPanel = () => {
  return (
    <div className="space-y-6">
      <h3 className="text-lg font-medium">System Information</h3>
      
      <Tabs defaultValue="architecture">
        <TabsList className="grid grid-cols-3">
          <TabsTrigger value="architecture">Architecture</TabsTrigger>
          <TabsTrigger value="process">Process Flow</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>
        
        <TabsContent value="architecture" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { 
                icon: <Layers className="h-5 w-5" />,
                title: "Document Chunking",
                description: "Research papers are split into semantic chunks with 150-200 token overlaps to maintain context."
              },
              {
                icon: <Database className="h-5 w-5" />,
                title: "Vector Database",
                description: "ChromaDB stores vector embeddings generated using sentence-transformers."
              },
              {
                icon: <Network className="h-5 w-5" />,
                title: "Knowledge Graph",
                description: "NetworkX-based graph storing entities and relationships extracted from the papers."
              },
              {
                icon: <Cpu className="h-5 w-5" />,
                title: "Language Model",
                description: "Uses large language models to generate answers from retrieved contexts."
              }
            ].map((item, index) => (
              <Card key={index}>
                <CardContent className="p-4 flex items-start gap-3">
                  <div className="bg-primary/10 rounded-full p-2 text-primary">
                    {item.icon}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">{item.title}</h4>
                    <p className="text-xs text-muted-foreground mt-1">
                      {item.description}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="process" className="mt-4">
          <Card>
            <CardContent className="p-4">
              <div className="space-y-6 py-2">
                {[
                  {
                    icon: <FileText />,
                    title: "Document Processing",
                    description: "Extract text from PDFs and split into overlapping chunks."
                  },
                  {
                    icon: <Database />,
                    title: "Embedding Generation",
                    description: "Convert text chunks to vector embeddings and store in ChromaDB."
                  },
                  {
                    icon: <Network />,
                    title: "Graph Construction",
                    description: "Extract entities and relationships to build a knowledge graph."
                  },
                  {
                    icon: <Loader2 />,
                    title: "Query Processing",
                    description: "Convert query to embedding and find similar chunks."
                  },
                  {
                    icon: <RefreshCw />,
                    title: "Graph Traversal",
                    description: "Enhance retrieval with graph-connected relevant chunks."
                  },
                  {
                    icon: <Cpu />,
                    title: "Answer Generation",
                    description: "Generate comprehensive answer from retrieved context."
                  }
                ].map((step, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <div className="bg-primary/10 rounded-full p-2 text-primary shrink-0">
                      {step.icon}
                    </div>
                    <div>
                      <h4 className="text-sm font-medium">{step.title}</h4>
                      <p className="text-xs text-muted-foreground mt-1">
                        {step.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="metrics" className="mt-4">
          <Card>
            <CardContent className="p-4">
              <div className="space-y-4">
                <p className="text-sm">
                  Performance metrics would be displayed here, including:
                </p>
                <ul className="list-disc pl-5 text-sm space-y-2">
                  <li>Retrieval precision and recall</li>
                  <li>Answer relevance scores</li>
                  <li>BLEU scores for answer quality</li>
                  <li>Processing times for each step</li>
                  <li>Knowledge graph statistics</li>
                </ul>
                <div className="mt-4">
                  <div className="flex items-center gap-2">
                    <LineChart className="h-5 w-5 text-primary" />
                    <span className="text-sm font-medium">Performance Visualization</span>
                  </div>
                  <div className="h-[200px] bg-secondary/50 rounded-md mt-2 flex items-center justify-center">
                    <p className="text-xs text-muted-foreground">
                      In a complete implementation, charts would display system performance metrics
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

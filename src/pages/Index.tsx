
import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { FileUploader } from "@/components/FileUploader";
import { PapersList } from "@/components/PapersList";
import { QueryResults } from "@/components/QueryResults";
import { KnowledgeGraph } from "@/components/KnowledgeGraph";
import { InfoPanel } from "@/components/InfoPanel";
import { useToast } from "@/components/ui/use-toast";
import { Info, Search, FileText, NetworkIcon } from "lucide-react";

const Index = () => {
  const [query, setQuery] = useState("");
  const [processing, setProcessing] = useState(false);
  const [uploadedPapers, setUploadedPapers] = useState<string[]>([]);
  const [answer, setAnswer] = useState("");
  const [retrievedChunks, setRetrievedChunks] = useState<string[]>([]);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const { toast } = useToast();

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) {
      toast({
        title: "Query is empty",
        description: "Please enter a question to search.",
        variant: "destructive",
      });
      return;
    }
    if (uploadedPapers.length === 0) {
      toast({
        title: "No papers uploaded",
        description: "Please upload at least one research paper before querying.",
        variant: "destructive",
      });
      return;
    }

    setProcessing(true);

    // In a real implementation, this would make API calls to your backend
    setTimeout(() => {
      // Simulate response from backend
      setAnswer("Based on the uploaded papers, the answer to your question is related to the knowledge graph enhancement of RAG systems. Research shows that incorporating knowledge graph relationships improves retrieval accuracy by 27% compared to vector search alone. The papers highlight that entity-relationship modeling provides crucial context that pure embedding-based approaches might miss.");
      
      setRetrievedChunks([
        "Knowledge graphs enhance RAG systems by providing structural relationships between entities that vector embeddings struggle to capture.",
        "Experimental results show a 27% improvement in retrieval accuracy when combining vector search with graph traversal algorithms.",
        "Entity disambiguation remains a challenge in knowledge graph construction, with current approaches achieving 83% accuracy on scientific text."
      ]);

      // Simulate knowledge graph data
      setGraphData({
        nodes: [
          { id: 1, label: "RAG Systems", group: 1 },
          { id: 2, label: "Knowledge Graphs", group: 1 },
          { id: 3, label: "Vector Embeddings", group: 2 },
          { id: 4, label: "Entity Extraction", group: 2 },
          { id: 5, label: "Retrieval Accuracy", group: 3 },
          { id: 6, label: "Context Enhancement", group: 3 },
        ],
        edges: [
          { from: 1, to: 2, label: "enhanced by" },
          { from: 2, to: 4, label: "requires" },
          { from: 1, to: 3, label: "uses" },
          { from: 2, to: 6, label: "provides" },
          { from: 3, to: 5, label: "impacts" },
          { from: 6, to: 5, label: "improves" },
        ]
      });

      setProcessing(false);
    }, 2000);
  };

  const handleFileUpload = (files: File[]) => {
    const newPapers = files.map(file => file.name);
    setUploadedPapers(prev => [...prev, ...newPapers]);
    
    toast({
      title: `${files.length} paper${files.length > 1 ? 's' : ''} uploaded`,
      description: "Papers have been added to your collection.",
    });
  };

  const handleRemovePaper = (paperToRemove: string) => {
    setUploadedPapers(prev => prev.filter(paper => paper !== paperToRemove));
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold tracking-tight mb-2">GraphRAG Q&A System</h1>
        <p className="text-lg text-muted-foreground">
          Knowledge graph-enhanced retrieval for research papers
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Paper Upload & Management */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center">
              <FileText className="mr-2 h-5 w-5" />
              Research Papers
            </CardTitle>
            <CardDescription>
              Upload PDF research papers to build your knowledge base
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FileUploader onFileUpload={handleFileUpload} />
            <div className="mt-4">
              <PapersList papers={uploadedPapers} onRemove={handleRemovePaper} />
            </div>
          </CardContent>
          <CardFooter className="flex justify-between text-sm text-muted-foreground">
            <span>{uploadedPapers.length} papers in collection</span>
          </CardFooter>
        </Card>

        {/* Center Column - Query Input & Results */}
        <Card className="col-span-1 lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Search className="mr-2 h-5 w-5" />
              Research Question
            </CardTitle>
            <CardDescription>
              Ask a question about the uploaded research papers
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleQuerySubmit} className="space-y-4">
              <Textarea 
                placeholder="Enter your research question here..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="min-h-[100px]"
              />
              <Button 
                type="submit" 
                className="w-full"
                disabled={processing || uploadedPapers.length === 0}
              >
                {processing ? "Processing..." : "Get Answer"}
              </Button>
            </form>

            {answer && (
              <div className="mt-6 space-y-4">
                <h3 className="text-lg font-semibold">Answer</h3>
                <div className="bg-secondary p-4 rounded-md">
                  {answer}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Results Section - Only shown after query */}
      {answer && (
        <div className="mt-8">
          <Tabs defaultValue="results">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="results">Retrieved Passages</TabsTrigger>
              <TabsTrigger value="graph">Knowledge Graph</TabsTrigger>
              <TabsTrigger value="info">System Information</TabsTrigger>
            </TabsList>
            <TabsContent value="results" className="p-4 border rounded-md mt-2">
              <QueryResults chunks={retrievedChunks} />
            </TabsContent>
            <TabsContent value="graph" className="p-4 border rounded-md mt-2">
              <KnowledgeGraph data={graphData} />
            </TabsContent>
            <TabsContent value="info" className="p-4 border rounded-md mt-2">
              <InfoPanel />
            </TabsContent>
          </Tabs>
        </div>
      )}

      {/* Information Alert - Shown when no papers are uploaded yet */}
      {uploadedPapers.length === 0 && (
        <Alert className="mt-8">
          <Info className="h-4 w-4" />
          <AlertTitle>Getting Started</AlertTitle>
          <AlertDescription>
            Upload research papers in PDF format to begin. The system will extract text, build a knowledge graph, 
            and enable semantic search with graph-enhanced retrieval.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default Index;

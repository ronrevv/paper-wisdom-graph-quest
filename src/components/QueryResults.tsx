
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

interface QueryResultsProps {
  chunks: string[];
}

export const QueryResults = ({ chunks }: QueryResultsProps) => {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium">Retrieved Passages</h3>
      <p className="text-sm text-muted-foreground mb-4">
        These passages were retrieved from the research papers based on your query and the knowledge graph.
      </p>
      
      <ScrollArea className="h-[400px]">
        <div className="space-y-3">
          {chunks.map((chunk, index) => (
            <Card key={index}>
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <div className="bg-primary/10 text-primary rounded-full h-6 w-6 flex items-center justify-center text-xs font-medium">
                    {index + 1}
                  </div>
                  <p className="text-sm">{chunk}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

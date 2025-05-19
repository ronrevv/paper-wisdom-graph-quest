
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface PapersListProps {
  papers: string[];
  onRemove: (paper: string) => void;
}

export const PapersList = ({ papers, onRemove }: PapersListProps) => {
  if (papers.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground text-sm">
        No papers uploaded yet
      </div>
    );
  }

  return (
    <ScrollArea className="max-h-[300px]">
      <ul className="space-y-2">
        {papers.map((paper, index) => (
          <li 
            key={paper + index}
            className="flex items-center justify-between p-2 rounded-md bg-secondary/50 hover:bg-secondary transition-colors"
          >
            <span className="text-sm truncate max-w-[200px]">{paper}</span>
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-6 w-6 text-muted-foreground hover:text-destructive"
              onClick={() => onRemove(paper)}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </li>
        ))}
      </ul>
    </ScrollArea>
  );
};

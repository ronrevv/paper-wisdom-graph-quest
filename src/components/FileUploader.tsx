
import { useState } from "react";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { cn } from "@/lib/utils";

interface FileUploaderProps {
  onFileUpload: (files: File[]) => void;
}

export const FileUploader = ({ onFileUpload }: FileUploaderProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const { toast } = useToast();
  
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
  };
  
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      processFiles(files);
    }
  };
  
  const processFiles = (files: File[]) => {
    const pdfFiles = files.filter(file => file.type === 'application/pdf');
    
    if (pdfFiles.length !== files.length) {
      toast({
        title: "Invalid file format",
        description: "Only PDF files are accepted",
        variant: "destructive",
      });
    }
    
    if (pdfFiles.length > 0) {
      onFileUpload(pdfFiles);
    }
  };
  
  return (
    <div
      className={cn(
        "border-2 border-dashed rounded-md p-6 text-center cursor-pointer transition-colors",
        isDragging ? "border-primary bg-muted" : "border-muted-foreground/25 hover:border-primary/50"
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        id="file-upload"
        type="file"
        multiple
        accept=".pdf"
        onChange={handleFileInputChange}
        className="hidden"
      />
      
      <label htmlFor="file-upload" className="cursor-pointer">
        <div className="flex flex-col items-center gap-2">
          <Upload className="h-8 w-8 text-muted-foreground" />
          <p className="text-sm font-medium">
            <span className="text-primary">Click to upload</span> or drag and drop
          </p>
          <p className="text-xs text-muted-foreground">
            PDF research papers only
          </p>
        </div>
      </label>
    </div>
  );
};

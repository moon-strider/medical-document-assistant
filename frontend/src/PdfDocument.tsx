import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import type { Span } from "./api";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

interface PdfDocumentProps {
  blobUrl: string;
  page: number;
  width: number;
  span: Span | null;
  bboxValid: boolean;
  sourceExcerpt: string;
  opening: string;
  onPageCount: (count: number) => void;
  onError: (message: string) => void;
}

export default function PdfDocument({
  blobUrl,
  page,
  width,
  span,
  bboxValid,
  sourceExcerpt,
  opening,
  onPageCount,
  onError,
}: PdfDocumentProps) {
  const box = span?.bbox;
  const pageWidth = span?.page_width;
  const pageHeight = span?.page_height;

  return (
    <div className="pdf-canvas">
      <Document
        file={blobUrl}
        onLoadSuccess={(document) => onPageCount(document.numPages)}
        onLoadError={(error) => onError(error.message)}
        loading={<div className="document-loading">{opening}…</div>}
      >
        <Page
          pageNumber={page}
          width={width}
          renderAnnotationLayer={false}
          renderTextLayer
        />
      </Document>
      {span?.page === page && bboxValid && box && pageWidth && pageHeight && (
        <div
          className="bbox-highlight"
          style={{
            left: `${(box[0] / pageWidth) * 100}%`,
            top: `${(box[1] / pageHeight) * 100}%`,
            width: `${((box[2] - box[0]) / pageWidth) * 100}%`,
            height: `${((box[3] - box[1]) / pageHeight) * 100}%`,
          }}
          aria-label={sourceExcerpt}
        />
      )}
    </div>
  );
}

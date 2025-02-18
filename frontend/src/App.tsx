import { useState, useEffect } from 'react'
import { debounce } from 'lodash-es'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2, Bold, Italic, Heading1, Heading2, List } from "lucide-react"
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import './components/editor.css'

interface Document {
  id: string
  title: string
  content: string
  created_at: string
  updated_at: string
  token?: string
}

function App() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [newTitle, setNewTitle] = useState('')
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null)
  const [loading, setLoading] = useState(true)
  const [connectionError, setConnectionError] = useState(false)
  const [saving, setSaving] = useState(false)
  const [downloading, setDownloading] = useState(false)

  const editor = useEditor({
    extensions: [StarterKit],
    content: selectedDoc?.content || '',
    onUpdate: debounce(async ({ editor }: { editor: any }) => {
      if (selectedDoc) {
        const content = editor.getHTML();
        setSaving(true);
        try {
          await fetch(`${import.meta.env.VITE_BACKEND_URL}/documents/${selectedDoc.id}`, {
            method: 'PUT',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              content: content
            })
          });
        } catch (error) {
          console.error('Failed to save document:', error);
        } finally {
          setSaving(false);
        }
      }
    }, 500), // 500ms debounce
  })

  useEffect(() => {
    fetchDocuments()
  }, [])

  useEffect(() => {
    if (editor && selectedDoc) {
      editor.commands.setContent(selectedDoc.content || '')
    }
  }, [selectedDoc, editor])

  const fetchDocuments = async () => {
    let retryCount = 0;
    const maxRetries = 3;
    
    while (retryCount < maxRetries) {
      try {
        setConnectionError(false);
        const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/documents`, {
          headers: {
            'Content-Type': 'application/json',
            'Accept-Encoding': 'gzip'
          }
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setDocuments(data);
        break;
      } catch (error) {
        retryCount++;
        if (retryCount === maxRetries) {
          console.error('Failed to fetch documents:', error);
          setConnectionError(true);
          alert('Failed to load documents. Please check your connection and refresh the page.');
        }
        await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
      } finally {
        setLoading(false);
      }
    }
  }

  const createDocument = async () => {
    if (!newTitle) return;
    try {
      setLoading(true);
      setConnectionError(false);
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/documents?title=${encodeURIComponent(newTitle)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const doc = await response.json();
      setDocuments([...documents, doc]);
      setNewTitle('');
    } catch (error) {
      console.error('Failed to create document:', error);
      setConnectionError(true);
      alert('Failed to create document. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  }

  const selectDocument = async (doc: Document) => {
    try {
      setLoading(true);
      setConnectionError(false);
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/documents/${doc.id}`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const updatedDoc = await response.json();
      setSelectedDoc(updatedDoc);
    } catch (error) {
      console.error('Failed to fetch document:', error);
      setConnectionError(true);
      alert('Failed to load document. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="container mx-auto p-4">
      <div className="flex flex-col space-y-4">
        {connectionError && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <strong className="font-bold">Connection Error!</strong>
            <span className="block sm:inline"> Unable to connect to the server. Please check your connection.</span>
            <button className="px-4 py-2 bg-red-500 text-white rounded mt-2" onClick={fetchDocuments}>
              Retry
            </button>
          </div>
        )}
        <Card>
          <CardHeader>
            <CardTitle>Professional Document Editor</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex space-x-2">
              <Input
                placeholder="Enter document title (e.g., My Professional Resume)"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && createDocument()}
              />
              <Button onClick={createDocument}>Create Document</Button>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>My Documents</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {documents.length === 0 ? (
                  <p className="text-muted-foreground text-center py-4">
                    No documents yet. Create one to get started!
                  </p>
                ) : (
                  documents.map((doc) => (
                    <Button
                      key={doc.id}
                      variant={selectedDoc?.id === doc.id ? "default" : "outline"}
                      className="w-full justify-start"
                      onClick={() => selectDocument(doc)}
                    >
                      {doc.title}
                    </Button>
                  ))
                )}
              </div>
            </CardContent>
          </Card>

          {selectedDoc && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div className="flex items-center gap-2">
                  <CardTitle>{selectedDoc.title}</CardTitle>
                  {saving && <Loader2 className="h-4 w-4 animate-spin" />}
                </div>
                <Button 
                  variant="outline"
                  onClick={async () => {
                    setDownloading(true);
                    try {
                      window.open(`${import.meta.env.VITE_BACKEND_URL}/documents/${selectedDoc.id}/download`, '_blank');
                    } finally {
                      setDownloading(false);
                    }
                  }}
                  disabled={downloading}
                >
                  {downloading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      Downloading...
                    </>
                  ) : (
                    'Download DOCX'
                  )}
                </Button>
              </CardHeader>
              <CardContent>
                <div className="min-h-[600px] flex flex-col space-y-2">
                  <div className="flex space-x-2 p-2 bg-muted rounded-lg">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => editor?.chain().focus().toggleBold().run()}
                      className={editor?.isActive('bold') ? 'bg-accent' : ''}
                    >
                      <Bold className="w-4 h-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => editor?.chain().focus().toggleItalic().run()}
                      className={editor?.isActive('italic') ? 'bg-accent' : ''}
                    >
                      <Italic className="w-4 h-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => editor?.chain().focus().toggleHeading({ level: 1 }).run()}
                      className={editor?.isActive('heading', { level: 1 }) ? 'bg-accent' : ''}
                    >
                      <Heading1 className="w-4 h-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => editor?.chain().focus().toggleHeading({ level: 2 }).run()}
                      className={editor?.isActive('heading', { level: 2 }) ? 'bg-accent' : ''}
                    >
                      <Heading2 className="w-4 h-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => editor?.chain().focus().toggleBulletList().run()}
                      className={editor?.isActive('bulletList') ? 'bg-accent' : ''}
                    >
                      <List className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="border rounded-lg overflow-hidden flex-1 bg-white p-4">
                    <EditorContent editor={editor} className="h-full" />
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

export default App

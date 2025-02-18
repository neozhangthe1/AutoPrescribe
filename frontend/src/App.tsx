import { useState, useEffect } from 'react'
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

  const editor = useEditor({
    extensions: [StarterKit],
    content: selectedDoc?.content || '',
    onUpdate: ({ editor }) => {
      if (selectedDoc) {
        const content = editor.getHTML();
        fetch(`${import.meta.env.VITE_BACKEND_URL}/documents/${selectedDoc.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            content: content
          })
        });
      }
    },
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
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/documents`, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setDocuments(data)
    } catch (error) {
      console.error('Failed to fetch documents:', error)
      alert('Failed to load documents. Please refresh the page.')
    } finally {
      setLoading(false)
    }
  }

  const createDocument = async () => {
    if (!newTitle) return
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/documents?title=${encodeURIComponent(newTitle)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const doc = await response.json()
      setDocuments([...documents, doc])
      setNewTitle('')
    } catch (error) {
      console.error('Failed to create document:', error)
      alert('Failed to create document. Please try again.')
    }
  }

  const selectDocument = async (doc: Document) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/documents/${doc.id}`, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const updatedDoc = await response.json()
      setSelectedDoc(updatedDoc)
    } catch (error) {
      console.error('Failed to fetch document:', error)
      alert('Failed to load document. Please try again.')
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
                <CardTitle>{selectedDoc.title}</CardTitle>
                <Button 
                  variant="outline"
                  onClick={() => window.open(`${import.meta.env.VITE_BACKEND_URL}/documents/${selectedDoc.id}/download`, '_blank')}
                >
                  Download DOCX
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

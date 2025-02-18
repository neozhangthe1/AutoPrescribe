import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2, Bold, Italic, Heading1, Heading2, List } from "lucide-react"
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import './components/editor.css'
import { config } from './config'

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
  const [generationQuery, setGenerationQuery] = useState('')
  const [previewContent, setPreviewContent] = useState('')
  const [optimizationRules, setOptimizationRules] = useState('')
  const [optimizedContent, setOptimizedContent] = useState('')

  const handlePreviewGeneration = () => {
    if (!generationQuery) return;
    
    const keywords = generationQuery.toLowerCase().split(',').map(k => k.trim());
    let content = '<h1>Professional Summary</h1>\n';
    content += '<p>Experienced professional with expertise in ' + keywords.join(', ') + '.</p>\n';
    content += '<h2>Key Skills</h2>\n<ul>\n';
    keywords.forEach(skill => {
      content += `<li>${skill.charAt(0).toUpperCase() + skill.slice(1)}</li>\n`;
    });
    content += '</ul>';
    
    setPreviewContent(content);
  }

  const handleApplyGeneration = () => {
    if (!previewContent || !editor) return;
    
    editor.commands.setContent(previewContent);
    setPreviewContent('');
    setGenerationQuery('');
  }

  const handlePreviewOptimization = () => {
    if (!optimizationRules || !editor) return;
    
    const currentContent = editor.getHTML();
    const optimized = optimizeContent(currentContent, optimizationRules);
    setOptimizedContent(optimized);
  }

  const handleApplyOptimization = () => {
    if (!optimizedContent || !editor) return;
    
    editor.commands.setContent(optimizedContent);
    setOptimizedContent('');
    setOptimizationRules('');
  }

  const optimizeContent = (content: string, rules: string) => {
    // Parse optimization rules
    const rulesList = rules.toLowerCase().split(',').map(r => r.trim());
    
    let optimized = content;
    
    // Apply professional tone rules
    if (rulesList.includes('professional tone')) {
      optimized = optimized.replace(/\b(good|nice|great)\b/gi, 'excellent');
      optimized = optimized.replace(/\b(big|huge|massive)\b/gi, 'significant');
      optimized = optimized.replace(/\b(got|grabbed|took)\b/gi, 'obtained');
    }
    
    // Apply action verbs rules
    if (rulesList.includes('action verbs')) {
      optimized = optimized.replace(/\b(did|made|created)\b/gi, 'implemented');
      optimized = optimized.replace(/\b(used|utilized)\b/gi, 'leveraged');
      optimized = optimized.replace(/\b(helped|assisted)\b/gi, 'facilitated');
    }
    
    // Apply formatting rules
    if (rulesList.includes('formatting')) {
      // Ensure proper spacing after periods
      optimized = optimized.replace(/\.(?! |\n|$)/g, '. ');
      // Remove double spaces
      optimized = optimized.replace(/  +/g, ' ');
    }
    
    return optimized;
  }

  const editor = useEditor({
    extensions: [StarterKit],
    content: selectedDoc?.content || '',
    onUpdate: ({ editor }) => {
      if (selectedDoc) {
        const content = editor.getHTML();
        console.log('Updating document:', selectedDoc.id);
        fetch(`${config.backendUrl}/documents/${selectedDoc.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({
            content: content
          })
        }).then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          console.log('Document updated successfully');
        }).catch(error => {
          console.error('Failed to update document:', error);
          alert('Failed to save changes. Please try again.');
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
    console.log('Fetching documents');
    try {
      const response = await fetch(`${config.backendUrl}/documents`, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      })
      console.log('Response status:', response.status);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('Fetched documents:', data);
      setDocuments(data);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
      alert('Failed to load documents. Please refresh the page.');
    } finally {
      setLoading(false);
    }
  }

  const createDocument = async () => {
    if (!newTitle) return;
    console.log('Creating document with title:', newTitle);
    try {
      const url = `${config.backendUrl}/documents?title=${encodeURIComponent(newTitle)}`;
      console.log('Making request to:', url);
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });
      console.log('Response status:', response.status);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const doc = await response.json();
      console.log('Created document:', doc);
      setDocuments([...documents, doc]);
      setNewTitle('');
    } catch (error) {
      console.error('Failed to create document:', error);
      alert('Failed to create document. Please try again.');
    }
  }

  const selectDocument = async (doc: Document) => {
    console.log('Selecting document:', doc.id);
    try {
      const response = await fetch(`${config.backendUrl}/documents/${doc.id}`, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });
      console.log('Response status:', response.status);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const updatedDoc = await response.json();
      console.log('Selected document:', updatedDoc);
      setSelectedDoc(updatedDoc);
    } catch (error) {
      console.error('Failed to fetch document:', error);
      alert('Failed to load document. Please try again.');
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

        <div className="grid grid-cols-1 gap-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Content Generation</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Input
                    placeholder="Enter keywords for content generation (e.g., software engineer, web development)"
                    value={generationQuery}
                    onChange={(e) => setGenerationQuery(e.target.value)}
                  />
                  <div className="flex space-x-2">
                    <Button onClick={handlePreviewGeneration}>Preview</Button>
                    <Button onClick={handleApplyGeneration} disabled={!previewContent}>Apply</Button>
                  </div>
                  {previewContent && (
                    <div className="p-4 border rounded bg-muted">
                      <p className="text-sm text-muted-foreground mb-2">Preview:</p>
                      <div dangerouslySetInnerHTML={{ __html: previewContent }} />
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Content Optimization</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Input
                    placeholder="Enter optimization rules (e.g., professional tone, action verbs)"
                    value={optimizationRules}
                    onChange={(e) => setOptimizationRules(e.target.value)}
                  />
                  <div className="flex space-x-2">
                    <Button onClick={handlePreviewOptimization}>Preview</Button>
                    <Button onClick={handleApplyOptimization} disabled={!optimizedContent}>Apply</Button>
                  </div>
                  {optimizedContent && (
                    <div className="p-4 border rounded bg-muted">
                      <p className="text-sm text-muted-foreground mb-2">Preview:</p>
                      <div dangerouslySetInnerHTML={{ __html: optimizedContent }} />
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

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
                  onClick={() => window.open(`${config.backendUrl}/documents/${selectedDoc.id}/download`, '_blank')}
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
    </div>
  );
}

export default App;

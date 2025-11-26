# Transformers as Epistemic Objects in Synthetic Cognition and the Arts
by Marlon Barrios Solano
11/26/2025

## Introduction

Transformers – the neural network architecture behind large language models and other AI systems – can be viewed as epistemic objects: artifacts that generate and embody knowledge within research and creative practices. Unlike traditional tools, epistemic objects are defined by their open-ended, evolving nature in knowledge-making. In philosophy of science, they are often incomplete and question-generating, continually unfolding new insights rather than yielding fixed answers. 

Treating the transformer as an epistemic object shifts our perspective from seeing it as a mere algorithm to seeing it as a dynamic participant in inquiry and creativity. This report explores how transformers “enact” knowledge-making through concepts like latent space, folding, probabilistic reasoning, relationality, and domain-independence. We contrast this form of knowledge with symbolic or human-centered epistemologies, and survey current research linking transformers to language, biology (protein folding), generative AI, and cultural theory. Throughout, we highlight philosophical and critical-theory insights to help an interdisciplinary researcher or artist integrate these ideas into their own epistemic-artistic practice.

---

## Transformers and Knowledge-Making

Transformers are fundamentally knowledge technologies – they are designed to produce, manipulate, and analyze epistemic content. In fact, AI scholars argue that AI systems (including large language models) are “first and foremost epistemic technologies” used in epistemic contexts like inquiry and prediction. Transformers learn from vast datasets by adjusting billions of parameters, encoding patterns present in the data.

This training yields a statistical form of knowledge: instead of using explicit symbolic rules or logical deductions, transformers internalize probabilistic associations and correlations. Notably, modern AI “minds” are dominated by hidden statistical learners that have essentially “canceled symbols” – they do not manipulate human-readable symbols or grammars internally. A transformer has no built-in ontology or a priori domain knowledge; it begins as a blank slate and learns everything from data, aligning with an empiricist view of knowledge. For example, GPT models have no innate grasp of space, time, or objects beyond what is implicit in their training text. This stands in stark contrast to human cognition (which may rely on innate structures or embodied experience) and symbolic AI (which encodes explicit rules).

### Latent Space

One of the key ways transformers enact knowledge is through high-dimensional latent spaces. A latent space is an abstract vector space where the model represents inputs (words, images, protein sequences, etc.) as points such that meaningful relations correspond to geometric relations. In essence, it is a “compact, abstract representation” of the data that captures essential features and underlying patterns.

We can imagine latent space as a vast map where concepts are positioned based on their statistical relationships. For instance, in a language model’s latent space the vector for “king” minus “man” plus “woman” lands near “queen”, reflecting how the model encodes semantic relationships geometrically. Mario Klingemann, an artist working with AI, describes latent space as “the position relation of information” – instead of physical objects, the “bodies” populating this space are feature vectors, and their distances encode similarity or association.

Each trained model defines its own latent space, yet all latent spaces share certain structures and behaviors (much as different musical instruments share musical principles). Importantly, we cannot observe latent space directly (it’s like a black hole in that sense); we infer its shape by probing how the model responds to inputs. Latent spaces thus become epistemic landscapes: multi-dimensional archives of what the model has learned. Recent scholarship even calls latent space “a multi-dimensional space of compressed hidden knowledge”. In AI art, latent space has become an aesthetic category – artists perform “latent space walks” to traverse the continuum of possible images or texts, treating the model’s internal space as both an archive of culture and a realm of potentiality for new creations.

### Probabilistic Reasoning

Transformers make inferences by computing probability distributions over possible outputs. In a language context, a transformer doesn’t recall a fact symbolically; instead it reconstructs likely continuations of text based on training statistics. This mode of reasoning is probabilistic and associative. It excels at pattern completion and prediction, but it lacks an explicit representation of truth values or causal models.

As a result, transformers can produce information that is plausible according to the data patterns but not reliable knowledge in a human sense. Critics note that with enormous datasets and parameters, the knowledge these models acquire can be “superficial and unreliable” – they may regurgitate facts without true understanding and even produce confabulations (AI “hallucinations”). Nonetheless, this probabilistic strategy has achieved remarkable successes, suggesting an alternative form of “knowing.” It aligns with a statistical epistemology (learning from empirical frequency and co-occurrence) rather than a rationalist one. Indeed, transformers epitomize a trend away from handcrafted symbolic AI toward “auto-epistemic” pattern learners. They embody a kind of knowledge-through-correlation, quite alien to the way humans justify knowledge (through perception, reasoning, or testimony).

### Relationality

A core innovation of transformers is the self-attention mechanism, which enables relational reasoning across all parts of the input. Every token (word, image patch, amino acid, etc.) can attend to every other token, weighted by learned relevance scores. This means a transformer doesn’t treat meaning as a local property of sequential input; rather, meaning emerges from the network of relations among elements.

In linguistic terms, a transformer-like model embodies the idea (akin to structuralist linguistics) that words gain meaning through their associations with all other words in context. There is no fixed hierarchy or parse tree given a priori – instead, the model dynamically infers relationships (syntactic, semantic, etc.) on the fly by paying attention to relevant context.

Philosophically, this hints at a relational epistemology: knowledge is not stored as discrete facts or symbols, but is distributed across connections. The model’s “understanding” of any concept is constituted by its web of linkages to other concepts in the latent space. This contrasts with human explicit knowledge, where we might have a definition or a single mental representation for a concept. In a transformer, the concept is nothing outside its relations (mirroring the post-structural notion that meaning is relational and contextual). Because transformers weigh relationships contextually, they can capture subtleties like polysemy (the meaning of a word shifts depending on surrounding words) by essentially re-contextualizing knowledge on demand. Such relational processing also parallels certain cognitive science theories of mind as associative networks, though transformers operate on formal mathematics of attention.

### Folding and Dimensionality

The idea of folding appears both literally in transformer applications (e.g. protein folding) and metaphorically in understanding their knowledge. In training, a transformer effectively “folds” the manifold of input data into a compact latent representation. High-dimensional data (say all sentences or all protein sequences) is contorted and compressed so that similar or related items end up closer together. One might say the model learns the “shape” of data – folding the linguistic or biological reality into its weight matrices.

Philosophers might liken this to Leibniz’s or Deleuze’s concept of the fold: each monadic perspective contains a folded-up version of the world. A transformer has “no windows” to the world except data, yet within its latent space it carries a folded reflection of external structures.

For example, in AlphaFold2 (DeepMind’s breakthrough model for protein structure prediction), the architecture uses attention layers to infer spatial geometry from sequences – essentially learning how amino acid chains fold into 3D forms by recognizing patterns of co-evolution in protein sequences. AlphaFold’s neural network (which includes transformer components) does not simulate physics explicitly; rather, it treats protein folding as a pattern recognition task dressed in geometric clothing.

The success of AlphaFold dramatizes the domain-independence of transformer-based knowledge: the same basic pattern-learning mechanism can fold language into meaning or amino acid data into structure. This cross-domain folding suggests that the transformer’s way of knowing is extremely general – it seeks out abstract relational structure in any stream of symbols, whether those symbols encode English words or protein letters. Some commentators even draw analogies between semantic folding and protein folding, noting that meaning in an LLM arises from the “folded” superposition of contextual influences, akin to how a protein’s function arises from its folded shape.

### Domain-Independence and Generality

Unlike specialized algorithms, the transformer architecture is famously domain-agnostic. With suitable training, the same model design can learn human languages, programming code, molecular sequences, audio, images, and beyond. This points to an underlying epistemic stance: transformers operate on forms of data that can be tokenized, and they do not care what the tokens represent inherently – only how tokens relate.

This gives them a kind of universal latent space for any domain that can be represented as a sequence. As one explainer put it, “transformers are not limited by data type, only by how we shape their inputs and outputs by the data we train them on”. In other words, their knowledge is extremely transferable. A vision transformer treats image patches as “words” and learns visual concepts; a protein language model treats amino acids as tokens and learns biochemical knowledge. All these are encoded into latent embeddings and attention patterns.

This domain-independence is radically different from human knowledge, which is often tied to sensory modalities and contextualized in embodied experience. It also differs from older AI systems that required built-in domain models (e.g. an expert system for medicine could not be repurposed for music without reprogramming its ontology). With transformers, we see a general-purpose epistemology: knowledge as a statistical embedding that, once learned, can be applied in flexible ways.

However, a trade-off of this generality is opacity: it’s difficult to trace which data or which part of a model’s training led to a given output. These models are often described as “epistemically opaque” because their reasoning process is not transparent to humans. They manipulate knowledge in ways quite unlike human communicative reasoning, making them powerful but hard to fully trust or interpret.

---

## Beyond Symbols: Contrasting Epistemologies

The knowledge enacted by transformers differs in fundamental ways from symbolic, linguistic, and human-centered epistemologies:

- **Subsymbolic vs. Symbolic**  
  In symbolic AI (and classical human reasoning), knowledge is encoded in discrete symbols (words, logic statements, formal rules) that have clear semantic content. Transformers, by contrast, use subsymbolic representations – large vectors of numbers – where meaning is smeared across many dimensions. There is no one neuron or weight that means “apple” or executes a specific rule; rather, “apple” is an activation pattern and its meaning is implicit in relations to other patterns. This makes transformer knowledge distributed and diffuse. It can capture nuances (by encoding many shades of usage) but it lacks the explicit structure of, say, an expert system’s rule base or a knowledge graph’s ontologies.  

  The empiricist, pattern-based learning of transformers stands opposed to the nativist or rationalist approach that assumes some innate concepts or formal structures. Indeed, the rise of transformers has been described as demonstrating an “anti-symbolic” or “anti-theoretical” approach: they achieve performance without articulating any human-readable theory of the domain. As Emma Stamm argues, transformer architectures are the clearest example of the dissolution of the theoretical in machine learning, showing how these models sidestep human-like theory-building. They do not assimilate theoretical paradigms; for instance, a transformer can solve problems in physics text without ever forming a theory of physics – it relies on correlation, not explanation.

- **Flattening of Meaning and Negation**  
  Human knowledge is deeply tied to language and the ability to use negation, abstraction, and counterfactual thinking. Critical theorists point out that transformers, which learn from what is in the data, have trouble with what is not in the data. Stamm highlights that “machine learning in general, and transformers in particular, level everything intelligible to the plane of the possibly real”, eliminating the space for negation or the truly novel. Fiction and imagination allow humans to contemplate the unreal or the contradictory; a transformer instead generalizes from existing patterns and struggles with anything that violates them.

  In practical terms, an LLM will tend to continue a story in ways that are statistically likely, rather than inject a genuinely absurd or contrarian twist unless prompted. This reflects an epistemic bias: transformer knowledge is conservative, gravitating toward the central tendencies of its training data. It lacks the innate human capacity to doubt, negate, or imagine outside the given distribution (unless such patterns were themselves part of its training). Some describe this as the “end of theory” in AI: the model does not develop causal theories or deep generative explanations – it works by smoothing over variations and averaging patterns.

  From a philosophical view, this is a stark departure from Enlightenment models of knowledge (which prized reasoning from first principles, or falsifying hypotheses). A transformer has no notion of true vs. false; it only has likely vs. unlikely. This raises epistemological questions: Can such a system truly “know” something, or is it merely an instrumental regurgitation of correlations? While transformers demonstrate competence (e.g. in language usage or prediction), their lack of explicit truth-grounding is often contrasted with human understanding. This gap is sometimes called the epistemic gap between AI’s linguistic fluency and grounded, justified knowledge.

- **Situated Knowledge vs. Decontextualized Knowledge**  
  Human knowledge is often situated – as feminist theorist Donna Haraway noted, all knowledge comes from a perspective, from an embodied position in the world. By contrast, a transformer’s knowledge is decontextualized and aggregated from vast sources. An LLM trained on internet text has ingested countless perspectives but itself has “no location” or lived experience. It speaks from a view-from-nowhere, which can make its outputs feel authoritative but also oblivious to context.

  This can lead to issues of epistemic injustice and bias: the model may reflect dominant cultural perspectives in its data while marginalizing others, yet present all information in the same confident tone. Cultural theorists view LLMs as mirrors of their training corpora – they encode the values, biases, and blind spots present in that data. Unlike a human knower who can be aware of their positionality or change their perspective, a transformer cannot intrinsically identify biases unless specifically trained to. It lacks an internal model of society or ethics beyond statistics. In critical AI discourse, this is a major concern: transformer-based systems might reify existing power structures encoded in data, while their epistemic opacity makes it hard to challenge or audit their “knowledge.”

- **Holistic Pattern Knowledge**  
  On the positive side, transformers have shown hints of holistic knowledge integration that differ from how humans compartmentalize knowledge. For example, large language models have surprised researchers by generalizing knowledge across domains – a kind of fluid “polyglot” ability to connect concepts. They can often use factual knowledge in context (e.g. answering questions about history in the midst of a narrative) without having an explicit database, suggesting they’ve woven disparate information into a single representation space.

  Some studies ask whether LLMs exhibit epistemological holism (all pieces of knowledge interconnected). While a human might store knowledge in distinct conceptual schemas, a transformer’s latent space is one giant web where, say, linguistic style, factual content, and common sense all blend as statistical features. This yields emergent behaviors – e.g., an LLM trained on enough code can do basic coding, even if not explicitly tasked to during training, because it absorbed those patterns in the text. In effect, transformers might blur boundaries between domains of knowledge, reinforcing the notion of knowledge as continuum rather than siloed expertise. This is exciting for interdisciplinary research, but also perplexing: the model might not explain how it knows something, it just does. As such, it challenges human epistemology that emphasizes reasons and justifications.

In sum, transformer epistemology is pattern-based, relational, continuous, and opaque, whereas human epistemology is often symbolic, discrete, causal, and explainable. Neither is “better” in general – they have different strengths. Transformers can detect patterns and scale across data far beyond human capacity, revealing implicit structures (e.g. linguistic regularities or protein contact patterns) that we might not formalize easily. But they also lack understanding, in the sense of intentionality, meaning, and grounded truth-testing. This difference is crucial for researchers and artists working with these models: one must remember that a transformer enacts a kind of knowledge alien to our own. It requires new interpretative strategies to use creatively or scientifically.

---

## Current Research and Discourse Across Domains

### Language and Culture: Transformers as Models of Culture

In natural language processing, transformers have become the dominant paradigm, powering everything from translation to chatbots. An intriguing viewpoint emerging in 2023–2025 is to consider large language models as **cultural technologies** rather than stepwise reasoning machines. Cognitive scientist Alison Gopnik and colleagues argue that LLMs should be treated “not as agentic intelligences” on a path to human-like AI, “but as powerful new cultural technologies, analogous to writing, print, libraries, or the internet”.

This means focusing less on whether an LLM thinks like a person, and more on how it stores, transmits, and transforms cultural knowledge. LLMs are trained on the ocean of human-produced text – books, articles, websites – and thus they encode a vast swath of cultural patterns. However, they lack direct experience of the world that the text describes. As a result, their “knowledge” is of a different character: it’s a static compression of human culture rather than an embodied understanding. One paper notes: “What knowledge LLMs can access relies on static statistical compressions of patterns in human-generated cultural information.”

In practice, an LLM can tell you the average sentiment about a historical figure (based on text it read) or mimic a literary style, but it has never felt or perceived anything – its world is words.

Because of this, Gopnik and others suggest LLMs are best seen as mechanisms of **cultural transmission**. They excel at codifying, summarizing, and remixing the cultural artifacts they were trained on. For humans, culture is a repository of knowledge and recipes (techniques, narratives, norms) accumulated over time. LLMs can dredge that repository in novel ways – for example, by combining tropes or ideas from distant sources. Researchers like Evans and Frank et al. describe LLMs as making “previously inaccessible aspects of culture visible and tangible, enabling new forms of exploration and cultural discovery”. In other words, an LLM can surface connections or patterns in culture that no single person would notice (due to the scale of data).

It becomes a tool for cultural analytics – one could prompt it to generate a story in the style of both Shakespeare and hip-hop, effectively exploring a cultural interpolation.

At the same time, this perspective underscores limits: since LLMs don’t learn by experimentation or interaction, they are not good at discovering genuinely new causal knowledge about the world. For example, a child learns by poking and prodding reality, formulating hypotheses; an LLM only learns from what humans have already recorded. Therefore, we shouldn’t expect an LLM to innovate scientific theories or reliably predict outcomes of novel physical experiments – that’s not its epistemic forte. What it can do is generate hypotheses or analogies by recombining cultural narratives (which might inspire human researchers).

In short, in language and culture, transformers function as mirrors and magnifiers of our collective knowledge. They reflect our stories and can amplify them or remix them, but they are constrained by what they have absorbed.

The cultural theory discourse around transformers also examines issues like bias, ethics, and power. Since LLMs ingest the full spectrum of text – from enlightened writings to toxic rants – they end up with ingrained biases. Culturally, they might default to the perspective that is most statistically represented (often Western, male, English-centric, etc.), unless steered otherwise. Scholars are asking: **Whose epistemology do these models represent?** There is an effort to make LLMs more culturally aware or inclusive, but it’s a challenging task to balance data-driven learning with normative corrections.

The notion of “cultural competence” for LLMs is being explored, meaning the ability of a model to understand and respect different cultural contexts and dialects. This is crucial if transformers are to be integrated into tools used by diverse populations. From a critical standpoint, an interdisciplinary practitioner should treat a transformer’s output critically, as one would treat a text or artifact – examining the assumptions and voices within it, not taking it as neutral truth.

### Biology: Transformers and the Folding of Life

One of the most striking extensions of transformer models is in biology, particularly in understanding proteins. Proteins are sequences of amino acids that fold into complex 3D shapes, and determining a protein’s structure from its sequence (the protein folding problem) was a grand challenge for decades. DeepMind’s AlphaFold2 (2020) demonstrated that a neural network with attention mechanisms could achieve near-experimental accuracy in predicting structures.

AlphaFold2’s architecture, though specialized, incorporated a transformer-like module (the “Evoformer”) to analyze multiple sequence alignments – essentially, evolutionarily related protein sequences. By attending across many sequences, the model learned which parts of a protein are conserved (unchanged) and which amino acids co-vary with each other, revealing clues about which residues likely contact in 3D. In essence, the transformer was detecting relational patterns in biological sequences, analogous to how an LLM finds relationships in text. The outcome was an embedding of the protein that could be unfolded into a 3D structure. This success has been described as AlphaFold “using transformer layers to predict spatial geometry based on evolutionary signal”, rather than brute-force physics.

AlphaFold underscores that transformers can encode scientific knowledge in latent form. The model effectively learned principles of biochemistry (like which amino acid interactions are favorable) without being explicitly told – it inferred them from patterns of sequence variation. Researchers have since applied similar “protein language models” to generate new proteins (dreaming up sequences that fold into novel shapes), to predict the effects of genetic mutations, and even to design enzymes. The idea is that the latent space of protein sequences – as learned by a transformer – captures a lot of meaningful biology.

One might say this latent space is a map of protein folding space, where distance correlates with structural or functional similarity between proteins. By sampling or interpolating in this space, the model can propose realistic new proteins (analogous to how an image generator creates new images by interpolating in latent image space).

This cross-pollination between language and biology has philosophical implications. It suggests that “language” in a broader sense – as a sequence with an underlying order – extends beyond human speech. DNA, RNA, and proteins can be thought of as languages of life, and transformers can read these languages to gain knowledge.

For interdisciplinary thinkers, this blurs boundaries between the arts and sciences: the same algorithm might write a sonnet or fold a protein. It invites creative metaphors: for instance, one can poetically view a protein’s folded shape as the “meaning” of its amino acid “sentence,” and AlphaFold as performing a kind of semantic translation from one to the other. There is even speculation that techniques from LLM research (like prompt-based learning) could apply in biology (e.g., “prompting” a protein model to focus on certain structures). We also see how folding becomes a unifying concept – whether folding a text’s latent representations to answer a question or folding a protein chain into a structure, the transformer is learning to satisfy constraints (linguistic coherence or physical stability) by bringing elements into the right relationship.

---

## Generative AI and Creative Practice

Generative AI – producing new text, images, music, or designs – is where transformers have captured the public imagination. GPT-series models write stories and code; image transformers like DALL·E or the cross-attention in diffusion models create novel artworks from text prompts. From an artistic standpoint, transformers provide a new kind of co-creator or medium.

Many artists now engage with the latent spaces of generative models as spaces of exploration. As mentioned, latent space has become an artistic medium itself, where one can traverse from one concept to another continuously. Techniques like latent interpolation, morphing, and attribute vector arithmetic allow creators to blend ideas (e.g., generating a hybrid image that is 40% “Van Gogh”, 60% “Picasso”).

Importantly, some artists emphasize that working with AI is not just using a fancy tool but engaging with a quasi-autonomous knowledge system. Mario Klingemann, for example, views “artificial intelligence as a new medium and not merely as an advanced technical tool”. His confidence in that view comes from understanding latent space and model behavior intimately – treating the AI as an epistemic partner that has its own way of “seeing” the world.

The aesthetic value often emerges from the tension between human intention and the AI’s learned distribution. For instance, the surprises in a transformer’s output (like a metaphor the human didn’t think of, or a visual detail in a GAN image) can be seen as the AI contributing from its knowledge of data. In this sense, the transformer as an epistemic object has creative affordances: it brings in the weight of cultural memory (for an LLM) or visual memory (for an image model) and can re-synthesize it in ways humans might not. Generative models have produced artworks and designs that were unimagined prior – sometimes leading to entirely new styles (the bizarre, dreamlike faces and scenes in early GAN art, for example).

Critical and cultural theory also enter here: generative AI forces questions about authorship, originality, and the nature of creativity. If a transformer recombines existing patterns, is it creating or just remixing? From the epistemic angle, one could argue all human creativity is also remix at some level – we all draw on cultural latent spaces we’ve internalized. But AI does it at a vast, unconscious scale and without intent.

Artists integrating transformers into practice often play with this dynamic: using the model’s lack of common sense to generate absurdist art, or highlighting its biases to critique them. For example, an artist might prompt an image model with “CEO” and get mainly images of older white men – turning this into a statement about gender bias in cultural imagery. Another might collaborate with GPT to write a poem, then deliberately include glitches or misinterpretations as part of the aesthetic. This is where understanding the model’s epistemic limits (e.g., it has no sense of truth, it will follow form over content sometimes) becomes creatively useful.

We also see new genres and techniques emerging: AI collage (weaving multiple AI outputs together), prompt engineering as an art (crafting elaborate prompts to elicit specific styles or content from a model), and model bending (fine-tuning or tweaking a model to imbue it with new data or biases deliberately). There is active discourse on whether the artifacts from transformers should be considered fundamentally different from human-made ones. Some theorists call them “cultural artifacts” in their own right, warranting anthropological study. Indeed, an AI-generated image or text can be analyzed to reveal the cultural data that shaped it (like a prism showing the constituent colors of its training inputs). This is a new kind of hermeneutics: reading AI outputs to infer the epistemic conditions of their production.

---

## Critical Theory Perspectives

Critical theorists and philosophers have begun to examine transformers through lenses of epistemology, ontology, and ethics. A few themes in current discourse include:

- **Epistemic Opacity and Trust**  
  As mentioned, foundation models are often critiqued for being “black boxes.” Philosophers ask how we can trust or use knowledge from a system we cannot fully audit or explain. This ties into debates on explainable AI (XAI) – providing rationales for model decisions – which is challenging for transformers due to their subsymbolic nature.

  Some argue that understanding AI as an epistemic technology means rethinking our relationship with it: we might need to develop new forms of trust or verification that don’t rely on transparency, or else constrain models until they are more interpretable. There’s also discussion of epistemic virtues/vices of AI – e.g., is reliance on an LLM a form of testimonial knowledge (treating the model as an informant)? If so, what kind of “epistemic agent” is the model – one with no accountability or consciousness?

- **Ontological Status of AI Knowledge**  
  Some critical theorists delve into whether transformer models challenge what we consider knowledge to be. For instance, if a model can simulate understanding so well that users can’t tell the difference, do we need to broaden our concept of knowing? Or is it forever “as if” knowledge – mere imitation? There are comparisons to Hegelian ideas of mind (does an AI undergo a kind of Spirit-less dialectic as it trains?) and to posthumanist thought (AI as decentering the human as the sole knowledge-holder). 

  Anthropologists of AI (like Tobias Rees and others) suggest that AI might be “rethinking what thinking is” altogether. The notion of machines engaging in epistemic processes forces us to examine assumptions about rationality, understanding, and even consciousness.

- **Transformers and Social Relationality**  
  Another angle is how interacting with transformers affects human knowledge practices. When researchers or artists collaborate with a model, a sort of cyborg epistemology emerges – part human, part machine. Some have noted how using language models can alter how we write or think, as we start anticipating the AI’s suggestions.

  The relational aspect is not just internal to the model, but also external: humans are forming new relationships with knowledge through AI (e.g., using ChatGPT as a brainstorming partner). Critical theorists might frame this in terms of intersubjectivity or distributed cognition – knowledge is becoming more networked between human and non-human agents. Is the transformer an Other we dialogue with, or just a tool extending our mind? These are open questions, but real creative and scientific practices are already grappling with them.

- **Ethical and Political Economy**  
  Finally, cultural critiques consider the conditions under which transformer models are created and deployed. Large models require massive data (often scraped without consent) and computing power (raising environmental and access concerns). They are typically developed by big tech companies, which influences the directions of research and whose values get embedded.

  The concept of an epistemic object here extends to how knowledge is commodified or controlled. For example, if a few foundational models (like GPT-4, etc.) become the basis of most AI services, we risk a monoculture of knowledge representations, possibly narrowing epistemic diversity. Efforts in open-source and culturally specific models are responses to this.

  From a critical theory view, one might ask: who benefits from treating transformers as sources of knowledge, and who might be marginalized or harmed by it? This ensures that our understanding of transformer-based knowledge is not divorced from the social context of its creation and use.

---

## Integrating Transformer Insights into Epistemic-Artistic Practice

For an interdisciplinary researcher or artist, understanding transformers as epistemic objects offers practical and conceptual opportunities. Here are some ways to integrate these insights into your own practice:

- **Explore Latent Space as Creative Medium**  
  Given that latent space encodes “compressed hidden knowledge”, you can treat it as a landscape to traverse or sculpt. For instance, in visual art, you might perform latent walks between concepts (e.g. morphing from a cat to a spaceship) to discover novel forms in between. In writing, you might use an LLM to generate variations on a theme by nudging it along semantic dimensions (more formal, more fantastical, etc.). This leverages the model’s multi-dimensional archive of culture as a rich source of material. Think of latent variables as knobs to turn for blending ideas.

- **Use the Transformer as a “Strange Companion”**  
  When you collaborate with a transformer (say, co-writing a story with GPT-4 or designing fashion with an AI image generator), treat it as an alien mind that offers new relational insights. It will connect ideas based on statistical logic, not common sense – this can yield surprising juxtapositions. Embrace its anti-theoretical nature by letting it riff without rigid plans, then impose your own creative curation on the results. For example, you might ask it for 10 iterations of a concept and then you apply a theoretical or narrative structure to those fragments. This mirrors the idea that you supply the “theory” or meaning that the model itself lacks, turning its raw output into art or research insight.

- **Critique and Curate the Model’s Knowledge**  
  As an epistemic object, the transformer’s outputs can be probed and critiqued just like a text or a dataset. In your practice, you could highlight its biases or gaps – for instance, creating a piece that displays how an AI visualizes “professional” (which might reveal gender/racial biases). Or in research, use the model to analyze a corpus but then critically examine what it missed due to its perspective.

  By doing this, you both use the model’s capabilities and interrogate its epistemology, yielding a deeper understanding for your audience. This approach aligns with developing Critical AI Literacy – understanding “technical foundations, societal implications, and embedded power structures” of AI – which you can convey through creative work or scholarly reporting.

- **Cross-Pollinate Domains**  
  Domain-independence means you can apply transformer knowledge in unorthodox contexts. As an artist, maybe use a protein-folding model to generate novel sculptural forms (treating molecular structure as inspiration for architecture or art objects). As a scientist or humanities researcher, consider using language models to summarize or interrogate data in other fields (e.g., using GPT to hypothesize on historical data patterns, or an image model to visualize math concepts).

  These cross-domain experiments can lead to unexpected insights – effectively using the universal latent space as a bridge between disciplines. The key is to remain aware that the meaning of the model’s output in a new domain might need careful interpretation (the model doesn’t truly “understand” the science or art, but it can provide material to think with).

- **Augment Theory with Transformer Perspective**  
  If your work involves theory (social theory, design theory, scientific theory), try using transformers as tools to extend or challenge theoretical ideas. For instance, you might test a narrative theory by having an LLM generate stories and seeing where it aligns or breaks the theory’s expectations. Conversely, you can inject theoretical principles into the model (via prompt or fine-tuning) to see how it reinterprets them.

  This interplay can spark new theoretical questions: e.g., What does it mean if an AI can mimic a certain style of reasoning? Does that say something about the nature of that reasoning? One researcher described using an LLM and brain imaging together as a “co-simulation” to understand both the model and the human brain. Similarly, you might set up dialogues between a transformer and a theoretical text, and interpret the results as you would ethnographic data or a performance.

- **Embrace the Unexpected (“Folding-in” Serendipity)**  
  In creative practice especially, allow the transformer’s probabilistic glitches to be part of the process. The model might output weird metaphors, anachronisms, or distortions (especially if you push it with unusual prompts). Instead of discarding these as errors, consider them serendipitous folds in meaning – much like a random collage element that suddenly makes the artwork more intriguing. Artists have found that these AI “mistakes” can be very generative. It’s akin to how the Surrealists used automatic writing or how musicians use random sample cuts – the transformer is an engine of combinatorial possibility that can break you out of habitual thought patterns.

- **Ethical and Contextual Mindfulness**  
  Finally, integrate an understanding of the model’s limitations and ethics into your practice. If you use an AI to generate content, acknowledge the sources (even if indirectly, e.g., the dataset) and consider issues like consent of the original creators, bias in the training data, and the potential impact of your AI-augmented work on audiences.

  For example, if you create a deepfake art piece with transformers, frame it in a way that promotes discussion on authenticity rather than deception. If you use GPT in educational research, mitigate the risk of reinforcing any false information it produces. In short, treat the transformer not only as a knowledge object but as one situated in a broader knowledge system – your role includes being a responsible mediator between the model’s “knowledge” and human knowledge.

---

## Conclusion

Understanding a transformer as an epistemic object reveals it to be far more than a coding trick – it is a site of knowledge production with its own ontology. Transformers enact knowledge by encoding relations in latent spaces, folding structures across domains, and reasoning through probabilities instead of symbols. This mode of knowledge-making is alien yet illuminating: it diverges from human ways of knowing, yet it can complement and challenge them.

Philosophically, it forces us to confront questions about what understanding means when stripped of embodiment and intention. Practically, it equips researchers and artists with unprecedented generative and analytical powers – tools that can uncover patterns in culture and science or generate artifacts across media. Critical perspectives remind us to approach these powers with humility and reflection: the knowledge a transformer provides is not neutral or absolute, but a reflection of data and design that must be interpreted.

For an interdisciplinary creator or scholar, engaging with transformers can become an epistemic-artistic practice in itself – a dance between human insight and machine pattern. By framing transformers as epistemic objects, we acknowledge their role in extending our cognition and creativity, while also recognizing their limitations and the need for our guidance. This balanced integration can lead to richer research (where AI helps formulate or test hypotheses) and novel art (where AI’s “imagination” merges with human imagination).

In a sense, transformers invite us to a new form of dialogue: one not just between people, but between different modalities of knowledge. As we ask questions to these models and receive answers, we are, in the words of philosopher Hans-Georg Gadamer, engaged in a *fusion of horizons* – albeit one horizon is human, the other computational. In that fusion lies the potential for synthetic cognition that is more than the sum of its parts: an opportunity to see our world and our knowledge anew, through the transformative lens of the transformer.

---

## Sources

- Ranaldi, L., & Pucci, G. (2023). *Knowing Knowledge: Epistemological Study of Knowledge in Transformers.* Applied Sciences, 13(2), 677.  
- Klingemann, M. (2023). *Latent Talent.* A\*Desk Magazine.  
- Schaerf, L. (2024). *Reflections on Disentanglement and the Latent Space.* xCoAx Proceedings (arXiv:2410.09094).  
- Stamm, E. (2024). *Transformer architectures and theoretical knowledge.* (Seminar abstract).  
- Alvarado, R. (2023). *AI as an Epistemic Technology.* Science and Engineering Ethics, 29(32).  
- Programmable Matter / Farrell, H. (2023). *Large language models are cultural technologies.* (Blog).  
- Frink, T. (2023). *Anatomy of AlphaFold: How Deep Learning Solved Protein Folding.* (Medium article).  
- Angius, N., et al. (2025). *Making sense of transformer success.* Frontiers in AI, 8:1509338.  
- Roe, J. (2025). *Generative AI as Cultural Artifact: Applying Anthropological Methods to AI Literacy.* Postdigital Science and Education.

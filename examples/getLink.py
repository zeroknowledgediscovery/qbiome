import os
import pygraphviz as pgv

# Directory containing the DOT files
directory = "./hypotheses_tmpdot_"

# Process each DOT file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".dot"):
        filepath = os.path.join(directory, filename)
        
        # Load the graph from the DOT file
        graph = pgv.AGraph(filepath)
        
        # Iterate through the nodes to find non-leaf nodes (nodes with outgoing edges)
        for node in graph.nodes():
            node_label = node.attr.get('label', node.get_name())
            if graph.out_degree(node) > 0:  # Check if node is a non-leaf node
                # Modify the node label to be a hyperlink
                new_label = f'<<a href="{node_label}.svg">{node_label}</a>>'
                node.attr['label'] = new_label
            else:
                # For leaf nodes, wrap the label in HTML-like syntax without changing the text
                new_label = f'<<table border="0" cellborder="0"><tr><td>{node_label}</td></tr></table>>'
                node.attr['label'] = new_label

        # Write the modified graph to a DOT file
        output_filepath = os.path.join(directory, f"modified_{filename}")
        graph.write(output_filepath)

        # Convert the modified DOT file to SVG using pygraphviz
        svg_output_filepath = output_filepath.replace('.dot', '.svg')
        graph.draw(svg_output_filepath, prog='dot', format='svg')

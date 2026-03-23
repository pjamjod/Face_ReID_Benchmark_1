from abc import ABC, abstractmethod
import pandas as pd
import os.path as osp
import json
import shutil
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import base64
from io import BytesIO
import imgkit
import markdown
import pdfkit

from ..utils.utils import clear_dir
from ..utils.params import Params

class GenericRevision:
    def __init__(self):
        self.params = Params()
        self.md_report = ''
        self.plots = {}
        self.tables = {}
        self.report_name = None
        pass

    @abstractmethod
    def load_params(self, params: dict):
        pass

    def save_tables_as_image(self, tables: dict, save_path: str, single_file: bool=False):
        if single_file:
            tables_html = {}
            for table_name, table in tables.items():
                tables_html[table_name] = self.table_to_html(table, table_name)    
            tables_html = '\n<br><br>\n'.join(list(tables_html.values()))
            imgkit.from_string(tables_html, osp.join(save_path, 'all_tables.png'), options={'format': 'png','quiet': '',})
        else:
            for table_name, table in tables.items():
                self.save_table_as_image(table, table_name, save_path)

    def save_table_as_image(self, table: pd.DataFrame, table_name: str, save_path: str):
        tables_html = self.table_to_html(table, table_name)
        imgkit.from_string(tables_html, osp.join(save_path, table_name+'.png'), options={'format': 'png','quiet': '',})
        
    def table_to_html(self, table: pd.DataFrame, table_name: str) -> str:
        html_table = table.to_html(index=False)
        table_title = '<caption style="font-weight: bold">'+table_name+'</caption>'
        html_lines = html_table.split('\n')
        html_lines.insert(1, table_title)
        html_table = '\n'.join(html_lines)
        return html_table 

    def tables_to_md(self, metrics: dict) -> dict:
        tables_dict = {}
        for metric_name, metric in metrics.items():
            tables_dict[metric_name] = self.table_to_md(metric, metric_name)
        return tables_dict

    def table_to_md(self, metric: pd.DataFrame, metric_name: str) -> str:
        metric_md = metric.to_markdown(index=False)
        md_text = '\n\n'+metric_md+'\n\n'+metric_name+'\n\n---\n\n'
        return md_text
    
    def generate_plots(self, plots_metrics: dict) -> dict:
        filtered_plots = self.merge_dict_by_root_key(plots_metrics)
        plots_dict = {}
        for plot_name, plot_data in filtered_plots.items():            
            for data_name, data in plot_data.items():
                columns = list(data.columns)
                x_column = columns[0]
                columns = columns[1:]
                if not columns is None:
                    break
            for column in columns:
                figure = plt.figure()
                ax = plt.gca()
                for data_name, data in plot_data.items():
                    if data.empty:
                        continue
                    epochs = data[x_column]
                    y = data[column]
                    ax.plot(epochs, y, label=data_name)
                ax.grid()
                ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
                plot_title = column.replace('_', ' ')
                ax.set_title(plot_title)
                ax.set_ylabel('_'.join(column.split('_')[1:]))
                ax.set_xlabel(x_column)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(10,len(epochs))))
                figure.tight_layout()
                plots_dict[column]=figure
                plt.close()

        return plots_dict
    
    def save_plots_as_image(self, plots_dict: dict, save_path: str, single_file: bool=False):
        if single_file:
            figure = self.merge_figures(plots_dict)
            self.save_plot_as_image(figure, save_path, 'all_plots')
        else:
            for plot_name, plot in plots_dict.items():
                self.save_plot_as_image(plot, save_path, plot_name)
    
    def save_plot_as_image(self, plot: plt.Figure, save_path: str, plot_name: str):
        plot.savefig(osp.join(save_path, plot_name+'.png'))
    
    def plots_to_base64(self, plots_dict: dict, plot_size: int=400):
        plots_base64 = {}
        for plot_name, plot in plots_dict.items():
            plots_base64[plot_name] = self.plot_to_base64(plot, plot_name, plot_size)
        return plots_base64
    
    def plot_to_base64(self, plot: plt.Figure, plot_title: str, plot_size: int=400) -> str:
        buf = BytesIO()
        plot.savefig(buf, format='png')
        buf.seek(0)
        png_data = buf.read()
        base64_utf8_str = base64.b64encode(png_data).decode('utf-8')
        dataurl = f'data:image/png;base64,{base64_utf8_str}'
        image_md = '<img src="'+dataurl+'" alt="'+plot_title+'" width="'+str(plot_size)+'"/>'
               
        return image_md

    def merge_figures(self, fig_dict: dict, max_cols: int=3) -> plt.Figure:
        num_figs = len(fig_dict)
        cols = min(max_cols, num_figs)  
        rows = (num_figs + cols - 1) // cols 
        figure, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows)) 
        axes = axes.flatten() if num_figs > 1 else [axes]
        for i, (key, fig) in enumerate(fig_dict.items()):
            for line in fig.gca().get_lines():
                axes[i].plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linestyle=line.get_linestyle())
            axes[i].set_title(fig.gca().get_title())
            axes[i].set_xlabel(fig.gca().get_xlabel())
            axes[i].set_ylabel(fig.gca().get_ylabel())
            axes[i].grid(fig.gca().get_xgridlines()[0].get_visible())
            legend = [x.get_text() for x in list(fig.gca().get_legend().get_texts())]
            axes[i].legend(legend, loc='upper left', bbox_to_anchor=(1, 0.5))
            axes[i].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(10,len(fig.gca().get_lines()[0].get_xdata()))))

        for i in range(num_figs, len(axes)):
            axes[i].axis('off')
    
        figure.tight_layout()

        return figure
    
    def save_report(self, save_path: str, save_plots: bool=True, save_tables: bool=True, single_file: bool=True, save_pdf: bool=False, save_html: bool=False):
        #clear_dir(save_path)
        os.makedirs(save_path, exist_ok=True)
        if save_plots:
            self.save_plots_as_image(self.plots, save_path, single_file=single_file)
        if save_tables:
            self.save_tables_as_image(self.tables, save_path, single_file=single_file)
        with open(osp.join(save_path, self.report_name+'.md'), 'w') as f:
            f.write(self.md_report)
        if save_pdf:
            self.save_md_as_pdf(save_path)
        if save_html:
            self.save_md_as_html(save_path)

    def md_to_html(self, md_text: str) -> str:
        html_content = markdown.markdown(md_text, extensions=['tables'])
        html_content = f"""
        <html>\n
        <head>
            <style>
                table, th, td {{
                    border: 1px solid black;
                    border-collapse: collapse;
                    padding: 5px;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>{html_content}</body>
        </html>
        """
        return html_content

    def save_md_as_html(self, save_path: str):
        html_content = self.md_to_html(self.md_report)
        with open(osp.join(save_path, self.report_name+'.html'), 'w') as f:
            f.write(html_content)
                    
    def save_md_as_pdf(self, save_path: str):
        html_content = self.md_to_html(self.md_report)
        pdf_file_path = osp.join(save_path, self.report_name+'.pdf')
        pdfkit.from_string(html_content, pdf_file_path)

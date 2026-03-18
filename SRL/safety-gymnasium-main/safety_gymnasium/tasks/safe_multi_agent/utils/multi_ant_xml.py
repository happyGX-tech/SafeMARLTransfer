from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np

import safety_gymnasium


def _map_suffix_token(token: str, agent_index: int) -> str:
    if token == 'agent1':
        return f'agent{agent_index}'
    if token == 'vision1':
        return f'vision{agent_index}'
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*1', token):
        return f'{token[:-1]}{agent_index}'
    return token


def _rename_elem_attrs(elem: ET.Element, agent_index: int, keep_childclass_agent1: bool = False) -> None:
    for key, value in list(elem.attrib.items()):
        if keep_childclass_agent1 and key == 'childclass':
            elem.attrib[key] = 'agent1'
            continue
        elem.attrib[key] = _map_suffix_token(value, agent_index)
    for child in list(elem):
        _rename_elem_attrs(child, agent_index, keep_childclass_agent1=False)


def _set_geom_rgba_recursive(elem: ET.Element, rgba: str) -> None:
    if elem.tag == 'geom':
        elem.attrib['rgba'] = rgba
    for child in list(elem):
        _set_geom_rgba_recursive(child, rgba)


def _agent_color_cycle() -> list[str]:
    return [
        '0.7412 0.0431 0.1843 1.0',
        '0.0039 0.1529 0.3961 1.0',
        '0.1686 0.5137 0.2588 1.0',
        '0.9725 0.6745 0.1098 1.0',
        '0.5804 0.4039 0.7412 1.0',
        '0.2275 0.5255 0.7686 1.0',
    ]


def ensure_multi_ant_xml(num_agents: int) -> str:
    if num_agents <= 2:
        return 'assets/xmls/multi_ant.xml'

    base_dir = os.path.dirname(safety_gymnasium.__file__)
    generated_rel_dir = os.path.join('tasks', 'safe_multi_agent', 'assets', 'xmls', 'generated')
    generated_abs_dir = os.path.join(base_dir, generated_rel_dir)
    os.makedirs(generated_abs_dir, exist_ok=True)

    output_filename = f'multi_ant_n{num_agents}.xml'
    output_abs_path = os.path.join(generated_abs_dir, output_filename)
    output_rel_path = os.path.join('assets', 'xmls', 'generated', output_filename)

    source_abs_path = os.path.join(base_dir, 'tasks', 'safe_multi_agent', 'assets', 'xmls', 'multi_ant.xml')
    tree = ET.parse(source_abs_path)
    root = tree.getroot()

    size = root.find('size')
    if size is not None:
        size.attrib['nconmax'] = str(max(1500, num_agents * 250))
        size.attrib['njmax'] = str(max(5000, num_agents * 900))

    worldbody = root.find('worldbody')
    if worldbody is None:
        raise RuntimeError('Invalid multi_ant.xml: worldbody not found')

    agent1_body = worldbody.find("body[@name='agent1']")
    if agent1_body is None:
        raise RuntimeError('Invalid multi_ant.xml: agent1 body not found')

    for idx in range(2, num_agents):
        body_clone = deepcopy(agent1_body)
        _rename_elem_attrs(body_clone, idx, keep_childclass_agent1=True)
        worldbody.append(body_clone)

    agent_bodies = []
    for idx in range(num_agents):
        name = 'agent' if idx == 0 else f'agent{idx}'
        body = worldbody.find(f"body[@name='{name}']")
        if body is not None:
            agent_bodies.append((idx, body))

    color_cycle = _agent_color_cycle()
    spawn_radius = max(0.8, 0.3 * num_agents)
    for idx, body in agent_bodies:
        theta = 2 * np.pi * idx / num_agents
        body.attrib['pos'] = f'{spawn_radius * np.cos(theta):.4f} {spawn_radius * np.sin(theta):.4f} 0.18'
        _set_geom_rgba_recursive(body, color_cycle[idx % len(color_cycle)])

    sensor = root.find('sensor')
    if sensor is None:
        raise RuntimeError('Invalid multi_ant.xml: sensor section not found')

    sensor_children = list(sensor)
    sensor_start = None
    for i, elem in enumerate(sensor_children):
        if elem.attrib.get('site') == 'agent1':
            sensor_start = i
            break
    if sensor_start is None:
        raise RuntimeError('Invalid multi_ant.xml: agent1 sensor chunk not found')

    sensor_chunk = sensor_children[sensor_start:]
    for idx in range(2, num_agents):
        for elem in sensor_chunk:
            elem_clone = deepcopy(elem)
            _rename_elem_attrs(elem_clone, idx)
            sensor.append(elem_clone)

    actuator = root.find('actuator')
    if actuator is None:
        raise RuntimeError('Invalid multi_ant.xml: actuator section not found')

    actuator_children = list(actuator)
    actuator_start = None
    for i, elem in enumerate(actuator_children):
        if elem.attrib.get('joint') == 'hip_11':
            actuator_start = i
            break
    if actuator_start is None:
        raise RuntimeError('Invalid multi_ant.xml: agent1 actuator chunk not found')

    actuator_chunk = actuator_children[actuator_start:]
    for idx in range(2, num_agents):
        for elem in actuator_chunk:
            elem_clone = deepcopy(elem)
            _rename_elem_attrs(elem_clone, idx)
            actuator.append(elem_clone)

    tree.write(output_abs_path, encoding='utf-8', xml_declaration=False)
    return output_rel_path.replace('\\', '/')
